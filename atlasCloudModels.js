import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MODEL_NODE_TYPES = ["AtlasCloudImage", "AtlasCloudVideo"];

// One catalogue request per (output, input) pair, shared by every node.
const catalogueCache = new Map();

async function fetchModels(outputKind, inputKind) {
    const key = `${outputKind}|${inputKind}`;
    if (catalogueCache.has(key)) return catalogueCache.get(key);
    const query = new URLSearchParams();
    if (outputKind) query.set("output", outputKind);
    if (inputKind && inputKind !== "Any") query.set("input", inputKind);
    const request = api
        .fetchApi(`/mdpack/atlas/models?${query.toString()}`)
        .then((response) => response.json())
        .then((body) => body.models || [])
        .catch((error) => {
            console.error("AtlasCloud: model list failed", error);
            return [];
        })
        .then((models) => {
            // Never cache a failed request or a genuinely empty result -- both
            // look identical to a caller (an empty array), and caching either
            // would leave a transient Atlas outage (or a momentarily-empty
            // filter) poisoning this (output, input) pair for the rest of the
            // session. Deleting the entry costs nothing when the result really
            // is empty (the next call just re-fetches the same empty answer),
            // but lets a real recovery show up on the very next interaction
            // instead of requiring a page reload.
            if (models.length === 0) catalogueCache.delete(key);
            return models;
        });
    catalogueCache.set(key, request);
    return request;
}

// Exposed for a later refresh-button task to bust the cache deliberately
// after POSTing /mdpack/atlas/refresh; this file adds no such button itself.
function clearCatalogueCache() {
    catalogueCache.clear();
}

function widgetOf(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

// Turn the plain `model` string widget into a combo without changing the
// serialised value: the workflow still stores the model id as a string, so a
// workflow submitted through the API stays valid.
// Turn the plain `model` string widget into a combo. Reassigning `.type` is
// not enough: the ComfyUI frontend instantiates a widget CLASS per type, so a
// widget declared as STRING stays a TextWidget -- with its own DOM element --
// and keeps rendering as a text field no matter what `.type` says. The widget
// object itself has to be replaced by a real combo instance.
//
// The replacement is spliced back in at the original index because ComfyUI
// serialises `widgets_values` positionally; appending it would shift every
// following widget in saved workflows. The serialised value stays a plain
// model-id string either way, so an API-submitted workflow remains valid.
function makeModelCombo(node) {
    const modelWidget = widgetOf(node, "model");
    if (!modelWidget) return null;
    if (modelWidget.constructor?.name === "ComboWidget" || modelWidget.isMDPackModelCombo) {
        return modelWidget;
    }

    const index = node.widgets.indexOf(modelWidget);
    const storedValue = modelWidget.value;
    const previousCallback = modelWidget.callback;

    // Drop the text widget together with the DOM node it owns, otherwise the
    // old input box keeps floating above the canvas.
    modelWidget.onRemove?.();
    modelWidget.element?.remove();
    node.widgets.splice(index, 1);

    const comboWidget = node.addWidget(
        "combo",
        "model",
        storedValue,
        (value) => previousCallback?.call(node, value),
        { values: [storedValue].filter(Boolean) },
    );
    comboWidget.isMDPackModelCombo = true;
    comboWidget.tooltip = modelWidget.tooltip;

    // addWidget appends; move it back to where the text widget was.
    node.widgets.splice(node.widgets.indexOf(comboWidget), 1);
    node.widgets.splice(index, 0, comboWidget);
    return comboWidget;
}

// Per-node "latest request wins" guard. onNodeCreated's initial refresh,
// onConfigure's post-load refresh, and rapid output_type/input_type changes
// can all have a fetchModels() call in flight for the same node at once;
// without this, whichever network response happens to resolve last would
// win, even if it belongs to a filter combination the node has since moved
// away from. Keyed by node via a WeakMap so it adds no serialized or
// user-visible state to the node itself.
const requestGenerations = new WeakMap();

async function refreshModelOptions(node, { keepValue = true } = {}) {
    const modelWidget = makeModelCombo(node);
    if (!modelWidget) return;
    const generation = (requestGenerations.get(node) || 0) + 1;
    requestGenerations.set(node, generation);

    const outputKind = widgetOf(node, "output_type")?.value || "";
    const inputKind = widgetOf(node, "input_type")?.value || "";
    const entries = await fetchModels(outputKind, inputKind);
    if (requestGenerations.get(node) !== generation) return; // superseded by a later refresh

    // An empty result -- a failed fetch (network/route down) or a filter that
    // genuinely matches nothing -- must never wipe a stored model id. Keep it
    // as the widget's only option so it stays visible until a real catalogue
    // comes back, instead of silently blanking a saved workflow's selection.
    if (entries.length === 0) {
        modelWidget.options.values = modelWidget.value ? [modelWidget.value] : [];
        modelWidget.options.labels = modelWidget.value ? [modelWidget.value] : [];
        node.graph?.setDirtyCanvas(true, true);
        return;
    }

    // Label carries the display name and the price, value stays the model id.
    modelWidget.options.values = entries.map((entry) => entry.model);
    modelWidget.options.labels = entries.map((entry) =>
        entry.price ? `${entry.display_name} — $${entry.price}` : entry.display_name
    );
    const stillListed = entries.some((entry) => entry.model === modelWidget.value);
    if (!keepValue) {
        // The user actively changed output_type/input_type: switch to the new
        // category's first entry.
        modelWidget.value = entries[0].model;
        modelWidget.callback?.(modelWidget.value);
    } else if (!stillListed && modelWidget.value) {
        // Workflow load (or any other keepValue refresh) named a model the
        // current filter excludes -- e.g. Atlas renamed it, or the stored
        // input/output kind no longer matches. Keep showing the stored model
        // rather than silently substituting one the user never chose; append
        // it so it stays selectable in the dropdown.
        modelWidget.options.values = [...modelWidget.options.values, modelWidget.value];
        modelWidget.options.labels = [...modelWidget.options.labels, modelWidget.value];
    }
    node.graph?.setDirtyCanvas(true, true);
}

// Guards against wrapping the output_type/input_type callbacks a second time
// if onNodeCreated ever fires again for the same node instance -- otherwise
// each further filter change would trigger one extra refreshModelOptions
// call per previous wrap (harmless with the generation guard above, since
// only the latest one can write back, but still a wasted request).
const filterCallbacksWrapped = new WeakSet();

// One schema request per model id, shared by every node that selects it.
const schemaCache = new Map();

async function fetchSchema(modelIdentifier) {
    if (schemaCache.has(modelIdentifier)) return schemaCache.get(modelIdentifier);
    const request = api
        .fetchApi(`/mdpack/atlas/schema?model=${encodeURIComponent(modelIdentifier)}`)
        .then((response) => response.json())
        .then((body) => (body && !body.error ? body : null))
        .catch((error) => {
            console.error("AtlasCloud: schema fetch failed", error);
            return null;
        })
        .then((schema) => {
            // A failed fetch (network error, or the route reporting an error --
            // e.g. Atlas unreachable) must not poison the cache: the next
            // selection of this model should retry instead of replaying the
            // same failure for the rest of the session. A schema that
            // genuinely declares no extra fields is a real, stable answer and
            // stays cached normally, same as an empty-but-successful catalogue
            // entry in fetchModels above.
            if (schema === null) schemaCache.delete(modelIdentifier);
            return schema;
        });
    schemaCache.set(modelIdentifier, request);
    return request;
}

// Exposed for tests and a later refresh-button task, mirroring clearCatalogueCache.
function clearSchemaCache() {
    schemaCache.clear();
}

const PARAM_PREFIX = "param_";

function readParams(node) {
    const paramsWidget = widgetOf(node, "params_json");
    try {
        const parsed = JSON.parse(paramsWidget?.value || "{}");
        return parsed && typeof parsed === "object" ? parsed : {};
    } catch (error) {
        return {};
    }
}

function writeParams(node) {
    const paramsWidget = widgetOf(node, "params_json");
    if (!paramsWidget) return;
    const values = {};
    for (const widget of node.widgets || []) {
        if (!widget.name.startsWith(PARAM_PREFIX)) continue;
        // Every parameter widget on screen is written, so what the node shows
        // is what the request carries. Withholding untouched defaults made the
        // two disagree: Seedream displayed size 2048*2048 while the request
        // carried none, and the model then derived its own size from the
        // reference images. See buildParamWidgets for how width/height from
        // older workflows are carried into the size widget instead.
        const value = widget.value;
        if (value === "" || value === null || value === undefined) continue;
        values[widget.name.slice(PARAM_PREFIX.length)] = value;
    }
    paramsWidget.value = JSON.stringify(values);
}

function removeParamWidgets(node) {
    node.widgets = (node.widgets || []).filter((widget) => !widget.name.startsWith(PARAM_PREFIX));
}

function widgetTypeFor(field) {
    if (Array.isArray(field.enum) && field.enum.length) return "combo";
    if (field.type === "boolean") return "toggle";
    if (field.type === "integer" || field.type === "number") return "number";
    return "text";
}

// Legacy widgets both nodes still declare for backward compatibility (a saved
// workflow keeps working even though a schema-driven param_<name> widget can
// now cover the same setting). Once the selected model's schema declares a
// field of the same name, the legacy widget is superseded -- showing both
// would let a user set the legacy one, watch it silently lose to params_json,
// and have no way to know why. A future node adding another such widget only
// needs to extend this list.
const LEGACY_WIDGET_NAMES = ["width", "height", "duration", "aspect_ratio", "resolution", "seed", "num_images"];

// Hide each legacy widget the selected model's schema has superseded, and
// restore the rest -- called on every schema application (including a model
// switch), so a widget hidden for one model reappears for the next one that
// needs it, the same reset discipline applyImageSlots already follows for
// the image slots.
function hideSupersededLegacyWidgets(node, schema) {
    const declaredFieldNames = new Set((schema.fields || []).map((field) => field.name));
    // width/height are a special case: a model with a `size` field derives
    // that string from them internally, so `size` supersedes both even
    // though the names don't match -- this cannot be derived generically.
    const hasSizeField = declaredFieldNames.has("size");
    for (const legacyName of LEGACY_WIDGET_NAMES) {
        const legacyWidget = widgetOf(node, legacyName);
        if (!legacyWidget) continue;
        const isDimension = legacyName === "width" || legacyName === "height";
        legacyWidget.hidden = declaredFieldNames.has(legacyName) || (isDimension && hasSizeField);
    }
}

// Rebuild the node's param_* widgets from a normalised schema. A value
// already present in params_json wins over the schema default -- this is how
// a saved workflow restores its chosen values, but it also applies when
// switching models, so a field name shared between the old and new model
// keeps the user's value. Fields left in params_json that have no matching
// field in the new schema are silently dropped once writeParams runs at the
// end: carrying a value forward from an incompatible model's schema (e.g. an
// enum choice that isn't valid here) would corrupt the next request rather
// than help the user.
// Turn the node's width/height widgets into the "WIDTH*HEIGHT" string a
// schema's `size` field expects, so dimensions set before this pack became
// schema-driven survive into the new widget instead of being replaced by the
// model's default. Returns undefined when there is nothing usable to carry
// over, or when the model does not offer that exact size.
function deriveSizeFromDimensions(node, sizeField) {
    const width = Number(widgetOf(node, "width")?.value) || 0;
    const height = Number(widgetOf(node, "height")?.value) || 0;
    if (width <= 0 || height <= 0) return undefined;
    const derived = `${width}*${height}`;
    const offered = sizeField.enum;
    if (Array.isArray(offered) && offered.length && !offered.includes(derived)) {
        console.warn(
            `AtlasCloud: this model does not offer ${derived}; leaving size at its default ` +
            `(${sizeField.default}). Pick one of: ${offered.join(", ")}`
        );
        return undefined;
    }
    return derived;
}

function buildParamWidgets(node, schema) {
    const storedValues = readParams(node);
    removeParamWidgets(node);
    for (const field of schema.fields || []) {
        const widgetName = `${PARAM_PREFIX}${field.name}`;
        const widgetKind = widgetTypeFor(field);
        const storedValue = storedValues[field.name];
        // A `size` field supersedes the node's width/height widgets, which are
        // hidden for exactly that reason. If a workflow still carries real
        // dimensions -- typically one saved before this pack became
        // schema-driven -- carry them into the size widget instead of letting
        // the schema default silently replace them. The derived value is only
        // used when the model actually offers it.
        const derivedSize = field.name === "size" ? deriveSizeFromDimensions(node, field) : undefined;
        const initialValue = storedValue !== undefined ? storedValue
            : derivedSize !== undefined ? derivedSize
            : field.default !== undefined && field.default !== null ? field.default
            : widgetKind === "number" ? 0 : "";
        // Everything the widget displays is also what gets sent. An earlier
        // revision withheld untouched schema defaults from params_json to stop
        // a model's `size` default from beating width/height -- but that made
        // the widget show a value the request never carried, and models like
        // Seedream do not re-apply their documented default server-side: with
        // no size they derive one from the reference images instead. Showing
        // one size and receiving another is the very failure this pack exists
        // to remove, so defaults are written, and the width/height collision is
        // handled by deriving from them above rather than by staying silent.
        const onChange = () => {
            widget.explicitValue = true;
            writeParams(node);
        };
        let widget;
        switch (widgetKind) {
            case "combo":
                widget = node.addWidget("combo", widgetName, initialValue, onChange,
                    { values: field.enum.slice() });
                break;
            case "toggle":
                widget = node.addWidget("toggle", widgetName, Boolean(initialValue), onChange);
                break;
            case "number":
                widget = node.addWidget("number", widgetName, Number(initialValue) || 0, onChange,
                    { step: field.type === "integer" ? 10 : 0.1,
                      precision: field.type === "integer" ? 0 : 2 });
                break;
            default:
                widget = node.addWidget("text", widgetName, String(initialValue), onChange);
        }
        // params_json is the single source of truth for these values; letting
        // ComfyUI also serialize the widgets themselves would create a second,
        // divergent copy the moment a model change adds/removes fields. Set on
        // both the widget and its options, matching keyValueDropdown.js's
        // mapping_toggle button.
        widget.serialize = false;
        widget.options = widget.options || {};
        widget.options.serialize = false;
        widget.tooltip = field.description || "";
        widget.explicitValue = storedValue !== undefined;
    }

    hideSupersededLegacyWidgets(node, schema);

    writeParams(node);
    if (typeof node.computeSize === "function" && typeof node.setSize === "function" && node.size) {
        const computedSize = node.computeSize();
        node.setSize([Math.max(node.size[0], computedSize[0]), computedSize[1]]);
    }
    node.graph?.setDirtyCanvas(true, true);
}

// Per-node "latest model wins" guard, mirroring requestGenerations above but
// kept as its own WeakMap since it tracks a different async operation:
// selecting model A then quickly model B must not let A's (now stale) schema
// response build widgets after B's schema has already been applied.
const schemaRequestGenerations = new WeakMap();

const IMAGE_SLOT_NAMES = ["image", "image_2", "image_3", "image_4", "image_5",
                          "image_6", "image_7", "image_8", "image_9", "image_10"];

function inputByName(node, name) {
    return (node.inputs || []).find((input) => input.name === name);
}

// Per-node cache of the image_fields array applyImageSlots last laid out,
// keyed by node. onConnectionsChange re-reads this to recompute input_map
// whenever the user connects or disconnects one of the ten image inputs,
// without a fresh schema fetch -- necessary because input_map's indices must
// track live connection state (see computeSingularInputMap below), not only
// whatever happened to be connected the moment the model was selected.
const nodeMediaFields = new WeakMap();

// atlascloud_api.py's collect_reference_images scans image..image_10 in slot
// order and skips every unconnected slot, so the reference list it hands to
// assign_media has no gaps: an image connected to image_3 while image_2 is
// empty lands at the same list index as if it had been connected to image_2
// instead. input_map's indices must therefore count *connected* slots, not
// declared slot positions -- otherwise a gap before a used slot shifts every
// later index, and assign_media either drops that reference (index out of
// range) or, worse, attaches it to the wrong field.
//
// This ranking assumes one reference image per connected slot, which holds
// for the ordinary case (a single loaded image wired into each slot) but not
// in general: collect_reference_images actually emits one reference *per
// image in the batch* (image_batch.shape[0]), so a slot fed by a multi-image
// batch contributes more than one entry to Python's list. If an earlier slot
// carries such a batch while a later slot is mapped to another singular
// field, the rank computed here and the index Python assigns will diverge --
// the browser has no visibility into tensor shapes, so this case cannot be
// corrected client-side. Task 10 adds a Python-side diagnostic that fires
// when input_map is in use and the collected reference count exceeds the
// number of connected slots, so the mismatch is surfaced instead of silently
// misrouting a frame.
function computeSingularInputMap(node, singularFields) {
    const connectedSlotRanks = new Map(); // slotIndex -> position in the compacted, gap-free list
    let rank = 0;
    IMAGE_SLOT_NAMES.forEach((slotName, slotIndex) => {
        const input = inputByName(node, slotName);
        if (input && input.link != null) {
            connectedSlotRanks.set(slotIndex, rank);
            rank += 1;
        }
    });
    const inputMap = {};
    for (const field of singularFields) {
        const connectedRank = connectedSlotRanks.get(field.slotIndex);
        // A field whose own slot isn't connected is left out of input_map
        // entirely (rather than mapped to some other image's index) so
        // assign_media simply leaves that payload field unset.
        if (connectedRank !== undefined) inputMap[field.name] = connectedRank;
    }
    return inputMap;
}

// ComfyUI validates connections against the node definition, so the slots
// always exist server-side; the UI only decides which ones are shown, how
// they are labelled, and (for models with several singular media fields)
// which schema field each connected slot feeds via input_map. Reads its
// layout from nodeMediaFields rather than taking a schema parameter directly,
// so it can be called again on a bare connection change (see
// onConnectionsChange below) without re-fetching anything.
function refreshImageSlots(node) {
    const mediaFields = nodeMediaFields.get(node) || [];
    const slotLabels = [];
    let inputMap = {};

    if (mediaFields.length === 1 && mediaFields[0].is_list) {
        const slotCount = Math.min(mediaFields[0].max_items || 10, IMAGE_SLOT_NAMES.length);
        for (let index = 0; index < slotCount; index += 1) {
            slotLabels.push(`${mediaFields[0].name} ${index + 1}`);
        }
        // A single list field takes every connected image in slot order, so
        // the Python default (no input_map) already does the right thing.
    } else if (mediaFields.length) {
        const singularFields = mediaFields.slice(0, IMAGE_SLOT_NAMES.length).map((field, index) => {
            slotLabels.push(field.name);
            return { name: field.name, slotIndex: index };
        });
        inputMap = computeSingularInputMap(node, singularFields);
    }

    IMAGE_SLOT_NAMES.forEach((slotName, index) => {
        const input = inputByName(node, slotName);
        if (!input) return;
        const isUsed = index < slotLabels.length;
        if (isUsed) {
            input.label = slotLabels[index];
            input.hidden = false;
        } else if (input.link != null) {
            // Never hide a slot that already has a link -- removing it would
            // silently break the user's graph.
            input.label = `${slotName} (unused by this model)`;
            input.hidden = false;
            console.warn(`AtlasCloud: ${slotName} is connected but the selected model ignores it`);
        } else {
            // Reset the label too, not just hidden: otherwise a label from a
            // previously selected, wider model would leak onto a slot this
            // model doesn't use next time it becomes visible.
            input.label = slotName;
            input.hidden = true;
        }
    });

    const inputMapWidget = widgetOf(node, "input_map");
    if (inputMapWidget) {
        inputMapWidget.value = Object.keys(inputMap).length ? JSON.stringify(inputMap) : "";
    }
    node.graph?.setDirtyCanvas(true, true);
}

// Both Atlas Cloud nodes only ever declare IMAGE inputs, so a video/audio
// media field (schema.image_fields entries carrying kind "video"/"audio")
// must never become a slot label or count toward the single-list-field rule
// below -- see the final review, Findings 4 and 5. Filtering here, before
// nodeMediaFields is ever read, also means a single image-kind list field
// gets the list treatment in refreshImageSlots even when a video/audio field
// sits right beside it in the schema (Finding 5's fix falls out of this for
// free: the mixed-field cases the review found were all exactly one
// image-kind field plus one video/audio field).
function imageKindFields(schema) {
    return (schema.image_fields || []).filter((field) => (field.kind || "image") === "image");
}

function applyImageSlots(node, schema) {
    nodeMediaFields.set(node, imageKindFields(schema));
    refreshImageSlots(node);
}

async function applySchema(node) {
    // Only nodes that carry a params_json widget can persist schema-driven
    // parameters -- both AtlasCloudImage and AtlasCloudVideo do. Without this
    // guard, selecting a model on a node with no params_json widget would
    // grow param_* widgets that writeParams can never save anywhere.
    if (!widgetOf(node, "params_json")) return;
    const modelIdentifier = widgetOf(node, "model")?.value;
    if (!modelIdentifier) return;
    const generation = (schemaRequestGenerations.get(node) || 0) + 1;
    schemaRequestGenerations.set(node, generation);
    const schema = await fetchSchema(modelIdentifier);
    if (schemaRequestGenerations.get(node) !== generation) return; // superseded by a later model selection
    if (schema === null) return; // fetch failed: leave existing widgets/params_json untouched
    buildParamWidgets(node, schema);
    applyImageSlots(node, schema);
    return schema;
}

app.registerExtension({
    name: "MDPack.AtlasCloudModels",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!MODEL_NODE_TYPES.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;
            if (!filterCallbacksWrapped.has(node)) {
                filterCallbacksWrapped.add(node);
                for (const filterName of ["output_type", "input_type"]) {
                    const filterWidget = widgetOf(node, filterName);
                    if (!filterWidget) continue;
                    const previousCallback = filterWidget.callback;
                    filterWidget.callback = function (value) {
                        const callbackResult = previousCallback?.apply(this, arguments);
                        refreshModelOptions(node, { keepValue: false });
                        return callbackResult;
                    };
                }

                const modelWidget = widgetOf(node, "model");
                if (modelWidget) {
                    const previousModelCallback = modelWidget.callback;
                    modelWidget.callback = function (value) {
                        const callbackResult = previousModelCallback?.apply(this, arguments);
                        applySchema(node);
                        return callbackResult;
                    };
                }
            }
            refreshModelOptions(node);
            applySchema(node);
            return result;
        };

        // input_map's indices depend on which image slots are actually
        // connected (see computeSingularInputMap), not only on which model is
        // selected -- so a link made or broken after the model was chosen
        // must recompute it too, without waiting for another schema fetch.
        // Only nodes applyImageSlots has already laid out once react here;
        // a node with no params_json widget never gets an entry in
        // nodeMediaFields and is left untouched (neither node type is in
        // that position today, but a future MODEL_NODE_TYPES addition could be).
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            if (nodeMediaFields.has(this)) refreshImageSlots(this);
            return result;
        };

        // Workflow load restores widget values after onNodeCreated, so refresh
        // the option list and rebuild the parameter widgets again, both while
        // keeping the values the workflow stored.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            refreshModelOptions(this, { keepValue: true });
            applySchema(this);
            return result;
        };
    },
});

export {
    MODEL_NODE_TYPES,
    fetchModels,
    refreshModelOptions,
    widgetOf,
    clearCatalogueCache,
    fetchSchema,
    clearSchemaCache,
    applySchema,
    buildParamWidgets,
    readParams,
    writeParams,
    widgetTypeFor,
    PARAM_PREFIX,
    LEGACY_WIDGET_NAMES,
    hideSupersededLegacyWidgets,
    IMAGE_SLOT_NAMES,
    applyImageSlots,
    refreshImageSlots,
};
