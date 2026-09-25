// Tests for the Atlas Cloud web extension (web/atlasCloudModels.js).
//
// Run with:
//   node test_atlascloud_web.mjs
//
// The extension imports "../../scripts/app.js" and "../../scripts/api.js",
// which only exist inside a running ComfyUI checkout. This suite reads the
// production file as text, rewrites those two import specifiers to point at
// stub modules written into a scratch temp directory, then dynamically
// imports the rewritten copy so the real exported functions (fetchModels,
// refreshModelOptions, widgetOf, MODEL_NODE_TYPES) can be exercised directly.
// The production file itself carries no test scaffolding.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const FAILURES = [];

function check(name, condition, detail = "") {
    if (condition) {
        console.log(`PASS ${name}`);
    } else {
        FAILURES.push(name);
        console.log(`FAIL ${name} ${detail}`);
    }
}

const THIS_DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
const SOURCE_PATH = path.join(THIS_DIRECTORY, "web", "atlasCloudModels.js");
const sourceText = fs.readFileSync(SOURCE_PATH, "utf-8");

const scratchDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "mdpack-atlas-web-"));

// --- stub scripts/app.js: only needs to accept the top-level registerExtension call ---
const stubAppPath = path.join(scratchDirectory, "stub-app.js");
fs.writeFileSync(
    stubAppPath,
    `export const app = {
        registerExtension(extension) {
            app.registeredExtension = extension;
        },
    };
    `,
    "utf-8"
);

// --- stub scripts/api.js: records every fetchApi URL and serves canned responses ---
const stubApiPath = path.join(scratchDirectory, "stub-api.js");
fs.writeFileSync(
    stubApiPath,
    `export const fetchCalls = [];
    export const fetchOptions = [];
    const responseMap = new Map();

    // Test hook: register the body handle_models would return for an exact URL.
    // Passing an Error makes fetchApi reject, simulating a network failure.
    // Passing a function defers resolution to whatever the function's
    // returned promise does -- used to control resolution order for the
    // out-of-order / race-condition test.
    export function setResponse(url, result) {
        responseMap.set(url, result);
    }

    export const api = {
        async fetchApi(url, options) {
            fetchCalls.push(url);
            fetchOptions.push(options);
            let result = responseMap.get(url);
            if (result === undefined) {
                throw new Error(\`test: no mock response configured for \${url}\`);
            }
            if (typeof result === "function") result = await result();
            if (result instanceof Error) throw result;
            return { json: async () => result };
        },
    };
    `,
    "utf-8"
);

const rewrittenSource = sourceText
    .replace(
        'import { app } from "../../scripts/app.js";',
        `import { app } from "${pathToFileURL(stubAppPath).href}";`
    )
    .replace(
        'import { api } from "../../scripts/api.js";',
        `import { api } from "${pathToFileURL(stubApiPath).href}";`
    );
check(
    "both production import statements were found and rewritten",
    rewrittenSource !== sourceText && !rewrittenSource.includes("../../scripts/"),
    "the source no longer matches the two known import lines -- update the rewrite above"
);

const rewrittenPath = path.join(scratchDirectory, "atlasCloudModels.js");
fs.writeFileSync(rewrittenPath, rewrittenSource, "utf-8");

const moduleUnderTest = await import(pathToFileURL(rewrittenPath).href);
const {
    fetchModels,
    refreshModelOptions,
    widgetOf,
    MODEL_NODE_TYPES,
    DEFAULT_PROVIDER,
    API_KEY_HEADER,
    providerOf,
    apiKeyOf,
    declaredWidgetsOf,
    requiredWidgetCountOf,
    migrateWidgetValues,
    fetchSchema,
    applySchema,
    buildParamWidgets,
    readParams,
    writeParams,
    widgetTypeFor,
    PARAM_PREFIX,
    LEGACY_WIDGET_NAMES,
    IMAGE_SLOT_NAMES,
    applyImageSlots,
    refreshImageSlots,
} = moduleUnderTest;
// Same absolute path as the rewritten import inside atlasCloudModels.js, so
// Node's module cache hands back the identical singleton -- fetchCalls/setResponse
// here observe exactly what the module under test's own `api` calls did.
const stubApiModule = await import(pathToFileURL(stubApiPath).href);
const { fetchCalls, fetchOptions, setResponse } = stubApiModule;

check(
    "MODEL_NODE_TYPES lists both Atlas Cloud node types",
    Array.isArray(MODEL_NODE_TYPES) &&
        MODEL_NODE_TYPES.includes("AtlasCloudImage") &&
        MODEL_NODE_TYPES.includes("AtlasCloudVideo"),
    JSON.stringify(MODEL_NODE_TYPES)
);

// --- widgetOf -----------------------------------------------------------------
{
    const node = { widgets: [{ name: "model", value: "x" }, { name: "output_type", value: "Image" }] };
    check("widgetOf finds a widget by name", widgetOf(node, "output_type")?.value === "Image");
    check("widgetOf returns undefined for a missing widget", widgetOf(node, "nope") === undefined);
    check("widgetOf tolerates a node with no widgets array", widgetOf({}, "model") === undefined);
}

// --- fetchModels: URL construction ---------------------------------------------
// Uses the (output=Image, input=Image) pair, which no other test below touches,
// so its cache entry cannot leak a stale response into the combo-value tests
// that fetch the (Image, Any) pair.
setResponse("/mdpack/atlas/models?output=Image&input=Image", { models: [] });
await fetchModels("Image", "Image");
check(
    "fetchModels includes both output and input parameters for a concrete input kind",
    fetchCalls.includes("/mdpack/atlas/models?output=Image&input=Image"),
    fetchCalls.join(", ")
);

// --- fetchModels: URL construction (Any) + catalogue cache ----------------------
// Both assertions share one (output=Text, input=Any) pair: the cache is keyed
// per pair and memoises the in-flight request, so a second call for the same
// pair must not be observable as a second entry in fetchCalls.
setResponse("/mdpack/atlas/models?output=Text", { models: [{ model: "a/text-model" }] });
const callsBeforeCachedPair = fetchCalls.length;
const firstCacheResult = await fetchModels("Text", "Any");
check(
    "fetchModels omits the input parameter when the kind is Any",
    fetchCalls.includes("/mdpack/atlas/models?output=Text"),
    fetchCalls.join(", ")
);
const callsAfterFirstCall = fetchCalls.length;
check(
    "the first call for a new (output, input) pair issues one network request",
    callsAfterFirstCall === callsBeforeCachedPair + 1,
    `before=${callsBeforeCachedPair} afterFirst=${callsAfterFirstCall}`
);
const secondCacheResult = await fetchModels("Text", "Any");
check(
    "a second call for the same pair issues no further request (cache hit)",
    fetchCalls.length === callsAfterFirstCall,
    `afterFirst=${callsAfterFirstCall} afterSecond=${fetchCalls.length}`
);
check(
    "the cached call returns the same model list",
    JSON.stringify(firstCacheResult) === JSON.stringify(secondCacheResult),
    JSON.stringify({ firstCacheResult, secondCacheResult })
);

// --- fetchModels: failed request ------------------------------------------------
setResponse("/mdpack/atlas/models?output=Audio", new Error("network down"));
const callsBeforeFailure = fetchCalls.length;
const failedResult = await fetchModels("Audio", "Any");
check(
    "a failed request resolves to an empty list instead of throwing",
    Array.isArray(failedResult) && failedResult.length === 0,
    JSON.stringify(failedResult)
);
check(
    "the failed request issued exactly one network call",
    fetchCalls.length === callsBeforeFailure + 1,
    `before=${callsBeforeFailure} after=${fetchCalls.length}`
);

// --- fetchModels: a failure is never cached, so the next call retries -----------
// Atlas recovering (or the ComfyUI route coming back) must be visible on the
// very next interaction, not only after a page reload -- so a failed result
// must not poison catalogueCache the way a real, non-empty catalogue does.
setResponse("/mdpack/atlas/models?output=Audio", { models: [{ model: "vendor/recovered-model" }] });
const retryResult = await fetchModels("Audio", "Any");
check(
    "a retry after a failed fetch issues a fresh network request instead of replaying the cached failure",
    fetchCalls.length === callsBeforeFailure + 2,
    `before=${callsBeforeFailure} after=${fetchCalls.length}`
);
check(
    "the retried request returns the newly available catalogue",
    retryResult.length === 1 && retryResult[0].model === "vendor/recovered-model",
    JSON.stringify(retryResult)
);

// --- refreshModelOptions: turns `model` into a combo with id values / labelled prices ---
// Mirrors how the ComfyUI frontend actually builds widgets: one CLASS per
// widget type. A STRING input is a TextWidget instance and stays one, which is
// why reassigning `.type` never turned the model field into a dropdown in the
// real UI -- the widget object has to be replaced. The stubs carry the same
// constructor names so the tests exercise that replacement for real.
class TextWidget {
    constructor({ name, value, callback, options }) {
        this.name = name;
        this.type = "string";
        this.value = value;
        this.callback = callback;
        this.options = options || {};
        this.removed = false;
    }
    onRemove() {
        this.removed = true;
    }
}

class ComboWidget {
    constructor({ name, value, callback, options }) {
        this.name = name;
        this.type = "combo";
        this.value = value;
        this.callback = callback;
        this.options = options || {};
    }
}

function addWidgetLikeComfyUI(node, type, name, value, callback, options) {
    const widget = type === "combo"
        ? new ComboWidget({ name, value, callback, options })
        : new TextWidget({ name, value, callback, options });
    node.widgets.push(widget);
    return widget;
}

function makeNode({ outputType = "Image", inputType = "Any", modelValue = "" } = {}) {
    const node = {
        widgets: [
            new TextWidget({ name: "model", value: modelValue, callback: undefined }),
            { name: "output_type", type: "combo", value: outputType },
            { name: "input_type", type: "combo", value: inputType },
        ],
    };
    node.addWidget = (type, name, value, callback, options) =>
        addWidgetLikeComfyUI(node, type, name, value, callback, options);
    return node;
}

const IMAGE_ENTRIES = [
    {
        model: "bytedance/seedream-v5.0-pro/text-to-image",
        display_name: "Seedream v5.0 Pro",
        price: "0.036",
    },
    {
        model: "black-forest-labs/flux-2-pro/text-to-image",
        display_name: "Flux 2 Pro",
        price: "0.05",
    },
];
setResponse("/mdpack/atlas/models?output=Image", { models: IMAGE_ENTRIES });

{
    const node = makeNode({ outputType: "Image", inputType: "Any", modelValue: "black-forest-labs/flux-2-pro/text-to-image" });
    await refreshModelOptions(node);
    const modelWidget = widgetOf(node, "model");
    check("refreshModelOptions converts the model widget to a combo", modelWidget.type === "combo");
    // Regression guard: the live ComfyUI frontend renders by widget CLASS, so
    // flipping `.type` on the TextWidget left the field rendering as a text box
    // even though every value underneath was correct. The widget object must be
    // an actual combo instance.
    check(
        "the model widget is a real combo instance, not a retyped text widget",
        modelWidget.constructor.name === "ComboWidget",
        modelWidget.constructor.name
    );
    check(
        "the replaced combo keeps the model widget's position, so widgets_values stays aligned",
        node.widgets.indexOf(modelWidget) === 0 && node.widgets.map((w) => w.name).join(",") === "model,output_type,input_type",
        node.widgets.map((w) => w.name).join(",")
    );
    check(
        "the discarded text widget was told to clean up its DOM element",
        node.widgets.every((w) => w.constructor.name !== "TextWidget" || w.name !== "model")
    );
    check(
        "combo values are the model ids, in catalogue order",
        JSON.stringify(modelWidget.options.values) === JSON.stringify(IMAGE_ENTRIES.map((e) => e.model)),
        JSON.stringify(modelWidget.options.values)
    );
    check(
        "combo labels carry the display name and price",
        modelWidget.options.labels[0] === "Seedream v5.0 Pro — $0.036" &&
            modelWidget.options.labels[1] === "Flux 2 Pro — $0.05",
        JSON.stringify(modelWidget.options.labels)
    );
    check(
        "a model already present in the filtered list survives the default keepValue refresh",
        modelWidget.value === "black-forest-labs/flux-2-pro/text-to-image",
        modelWidget.value
    );
}

// --- refreshModelOptions: output_type change re-filters and drops the stale model ---
const VIDEO_ENTRIES = [
    { model: "kling/v2.0", display_name: "Kling v2.0", price: "0.3" },
    { model: "google/veo-3.1", display_name: "Veo 3.1", price: "0.4" },
];
setResponse("/mdpack/atlas/models?output=Video", { models: VIDEO_ENTRIES });

{
    const node = makeNode({ outputType: "Image", inputType: "Any", modelValue: "bytedance/seedream-v5.0-pro/text-to-image" });
    await refreshModelOptions(node); // populate for Image, keepValue defaults true
    const modelWidget = widgetOf(node, "model");
    check(
        "sanity: the Image-filtered model is kept before switching output_type",
        modelWidget.value === "bytedance/seedream-v5.0-pro/text-to-image"
    );

    // Simulate the wrapped output_type callback: the widget's own value changes,
    // then the extension re-filters with keepValue: false.
    widgetOf(node, "output_type").value = "Video";
    await refreshModelOptions(node, { keepValue: false });
    check(
        "switching output_type re-filters the model list to the new kind",
        JSON.stringify(modelWidget.options.values) === JSON.stringify(VIDEO_ENTRIES.map((e) => e.model)),
        JSON.stringify(modelWidget.options.values)
    );
    check(
        "the stale model (not in the new list) is replaced by the first entry of the new list",
        modelWidget.value === VIDEO_ENTRIES[0].model,
        modelWidget.value
    );
}

// --- refreshModelOptions: empty result never destroys a stored model id ---------
// Whether the catalogue is genuinely empty for this filter or the fetch
// failed, the widget must not crash and must not blank a value the user (or
// a saved workflow) already had selected.
setResponse("/mdpack/atlas/models?output=3D", { models: [] });

{
    const node = makeNode({ outputType: "3D", inputType: "Any", modelValue: "some/stale-model" });
    await refreshModelOptions(node);
    const modelWidget = widgetOf(node, "model");
    check(
        "an empty catalogue result keeps a previously stored model value instead of clearing it",
        modelWidget.value === "some/stale-model",
        modelWidget.value
    );
    check(
        "the stored model stays the widget's sole listed option",
        JSON.stringify(modelWidget.options.values) === JSON.stringify(["some/stale-model"]),
        JSON.stringify(modelWidget.options.values)
    );
}

{
    const node = makeNode({ outputType: "3D", inputType: "Any", modelValue: "" });
    await refreshModelOptions(node);
    const modelWidget = widgetOf(node, "model");
    check("an empty catalogue result with no stored value leaves the value empty", modelWidget.value === "");
    check(
        "an empty catalogue result with no stored value leaves an empty values list",
        modelWidget.options.values.length === 0
    );
}

// --- refreshModelOptions: failed fetch (network/route down) keeps the stored model ---
// Uses a pair not touched anywhere else in this file: a failed fetch is no
// longer cached (see the catalogueCache fix above), so reusing the Audio/Any
// pair from the fetchModels-level failure test above would not observe a
// failure here -- that pair's cache entry was replaced by a real, cached
// success in the "retry after a failed fetch" test that ran after it.
setResponse("/mdpack/atlas/models?output=Text&input=Video", new Error("route unreachable"));

{
    const node = makeNode({ outputType: "Text", inputType: "Video", modelValue: "vendor/still-selected-model" });
    await refreshModelOptions(node);
    const modelWidget = widgetOf(node, "model");
    check(
        "a failed fetch (catalogue/route unreachable) leaves a previously stored value intact",
        modelWidget.value === "vendor/still-selected-model",
        modelWidget.value
    );
    check(
        "a failed fetch leaves the stored value as the widget's sole listed option",
        JSON.stringify(modelWidget.options.values) === JSON.stringify(["vendor/still-selected-model"]),
        JSON.stringify(modelWidget.options.values)
    );
}

// --- refreshModelOptions: keepValue true with a stored model absent from a non-empty list ---
// The saved-workflow case: the current filter's catalogue is real and
// non-empty, but does not include the model the workflow stored (renamed,
// or the filter no longer matches). The stored model must stay selected and
// stay visible in the dropdown rather than being swapped for a model the
// user never chose.
const IMAGE_INPUT_VIDEO_ENTRIES = [{ model: "vendor/only-listed-model", display_name: "Only Listed Model", price: "0.02" }];
setResponse("/mdpack/atlas/models?output=Image&input=Video", { models: IMAGE_INPUT_VIDEO_ENTRIES });

{
    const node = makeNode({ outputType: "Image", inputType: "Video", modelValue: "vendor/renamed-or-missing-model" });
    await refreshModelOptions(node); // keepValue defaults to true, mirroring onConfigure
    const modelWidget = widgetOf(node, "model");
    check(
        "keepValue:true with a stored model missing from a non-empty filtered list keeps that value",
        modelWidget.value === "vendor/renamed-or-missing-model",
        modelWidget.value
    );
    check(
        "the stored model is appended to options.values so it stays selectable",
        JSON.stringify(modelWidget.options.values) ===
            JSON.stringify(["vendor/only-listed-model", "vendor/renamed-or-missing-model"]),
        JSON.stringify(modelWidget.options.values)
    );
    check(
        "the stored model is appended to options.labels too, so values and labels stay parallel arrays",
        JSON.stringify(modelWidget.options.labels) ===
            JSON.stringify(["Only Listed Model — $0.02", "vendor/renamed-or-missing-model"]),
        JSON.stringify(modelWidget.options.labels)
    );
}

// --- refreshModelOptions: out-of-order resolution does not let a stale request win ---
// The regression this guards against: the user switches output_type from A to
// B before A's slow response comes back. Without a "latest request wins"
// guard, A's response could resolve after B's and overwrite the widget with
// A's (now stale) data even though the node has moved on to B.
function createDeferred() {
    let resolve;
    const promise = new Promise((res) => {
        resolve = res;
    });
    return { promise, resolve };
}

const RACE_A_ENTRIES = [{ model: "vendor/race-a-model", display_name: "Race A Model", price: "0.01" }];
const RACE_B_ENTRIES = [{ model: "vendor/race-b-model", display_name: "Race B Model", price: "0.02" }];
const deferredA = createDeferred();
const deferredB = createDeferred();
setResponse("/mdpack/atlas/models?output=RaceA", () => deferredA.promise);
setResponse("/mdpack/atlas/models?output=RaceB", () => deferredB.promise);

{
    const node = makeNode({ outputType: "RaceA", inputType: "Any", modelValue: "" });
    const slowCallForA = refreshModelOptions(node); // starts fetching RaceA, not yet resolved

    // Simulate the user changing output_type before A's response arrives.
    widgetOf(node, "output_type").value = "RaceB";
    const fastCallForB = refreshModelOptions(node, { keepValue: false }); // starts fetching RaceB

    // B answers first (fast reply); A answers after (slow, and by now stale).
    deferredB.resolve({ models: RACE_B_ENTRIES });
    await fastCallForB;
    deferredA.resolve({ models: RACE_A_ENTRIES });
    await slowCallForA;

    const modelWidget = widgetOf(node, "model");
    check(
        "a stale, later-resolving response for the abandoned filter does not overwrite the widget",
        JSON.stringify(modelWidget.options.values) === JSON.stringify(RACE_B_ENTRIES.map((e) => e.model)) &&
            modelWidget.value === RACE_B_ENTRIES[0].model,
        `values=${JSON.stringify(modelWidget.options.values)} value=${modelWidget.value}`
    );
}

// =================================================================================
// Task 8: schema-driven parameter widgets
// =================================================================================

// A minimal node fixture with a working addWidget mock, since buildParamWidgets/
// applySchema need to actually create widgets (unlike makeNode above, which only
// needs a plain widgets array for refreshModelOptions).
function makeParamNode({ modelValue = "vendor/model-a", paramsJsonValue = "{}", includeWidthHeight = true, includeVideoLegacyWidgets = true, includeSeedAndNumImages = true } = {}) {
    const node = {
        widgets: [
            { name: "model", type: "combo", value: modelValue, callback: undefined, options: { values: [modelValue] } },
            { name: "params_json", type: "string", value: paramsJsonValue },
        ],
        graph: { setDirtyCanvas() {} },
        computeSize: () => [220, 120],
        setSize() {},
        size: [220, 120],
    };
    if (includeWidthHeight) {
        node.widgets.push({ name: "width", value: 0, hidden: false });
        node.widgets.push({ name: "height", value: 0, hidden: false });
    }
    if (includeVideoLegacyWidgets) {
        node.widgets.push({ name: "duration", value: 0, hidden: false });
        node.widgets.push({ name: "aspect_ratio", value: "", hidden: false });
        node.widgets.push({ name: "resolution", value: "", hidden: false });
    }
    if (includeSeedAndNumImages) {
        node.widgets.push({ name: "seed", value: -1, hidden: false });
        node.widgets.push({ name: "num_images", value: 1, hidden: false });
    }
    node.addWidget = (type, name, value, callback, options) => {
        const widget = { type, name, value, callback, options: options ? { ...options } : undefined };
        node.widgets.push(widget);
        return widget;
    };
    return node;
}

// --- buildParamWidgets: widget type mapping -------------------------------------
{
    const schema = {
        fields: [
            { name: "size", type: "string", default: "2048*2048", enum: ["2048*2048", "1328*1776"], description: "Output size" },
        ],
    };
    const node = makeParamNode();
    buildParamWidgets(node, schema);
    const sizeWidget = widgetOf(node, "param_size");
    check("an enum field produces a combo widget", sizeWidget?.type === "combo", sizeWidget?.type);
    check(
        "the combo widget's values are the schema's enum list",
        JSON.stringify(sizeWidget?.options?.values) === JSON.stringify(["2048*2048", "1328*1776"]),
        JSON.stringify(sizeWidget?.options?.values)
    );
    check("the combo widget defaults to the schema default", sizeWidget?.value === "2048*2048", sizeWidget?.value);
    check(
        "the dynamic widget itself is excluded from ComfyUI's own serialisation (params_json is the single source of truth)",
        sizeWidget?.serialize === false && sizeWidget?.options?.serialize === false,
        `serialize=${sizeWidget?.serialize} options.serialize=${sizeWidget?.options?.serialize}`
    );
}

{
    const schema = {
        fields: [
            { name: "thinking", type: "boolean", default: true },
            { name: "duration", type: "integer", default: 5 },
            { name: "guidance_scale", type: "number", default: 7.5 },
            { name: "negative_prompt", type: "string", default: "" },
        ],
    };
    const node = makeParamNode();
    buildParamWidgets(node, schema);
    check("a boolean field produces a toggle widget", widgetOf(node, "param_thinking")?.type === "toggle");
    check("an integer field produces a number widget", widgetOf(node, "param_duration")?.type === "number");
    check("a number field produces a number widget", widgetOf(node, "param_guidance_scale")?.type === "number");
    check("a plain string field produces a text widget", widgetOf(node, "param_negative_prompt")?.type === "text");
}

// --- buildParamWidgets / writeParams: params_json serialisation -----------------
{
    const schema = {
        fields: [
            { name: "size", type: "string", enum: ["a", "b"], default: "a" },
            { name: "negative_prompt", type: "string", default: "" },
        ],
    };
    const node = makeParamNode();
    buildParamWidgets(node, schema);
    check(
        "building the widgets omits a field left at its empty-string default",
        readParams(node).negative_prompt === undefined,
        JSON.stringify(readParams(node))
    );
    // What the widget shows is what the request carries. Withholding untouched
    // defaults (an earlier revision did) made the node display Seedream's
    // size 2048*2048 while sending no size at all, and the model then derived
    // its own from the reference images -- a shown value that never reached
    // the API.
    check(
        "a field left at its schema default is still written, so shown and sent agree",
        readParams(node).size === "a",
        JSON.stringify(readParams(node))
    );
    check(
        "the widget still displays the schema default so the UI looks right",
        widgetOf(node, "param_size")?.value === "a",
        widgetOf(node, "param_size")?.value
    );

    const sizeWidget = widgetOf(node, "param_size");
    sizeWidget.value = "b";
    sizeWidget.callback();
    check("changing a widget rewrites params_json as a JSON object with the new value", readParams(node).size === "b");
}

// --- Fix 2: a freshly selected model's untouched defaults never populate
// params_json, so the node's own width/height widgets stay free to convert
// (the width/height <-> size interaction itself is proven end-to-end on the
// Python side, in test_atlascloud_api.py and test_atlascloud_models.py) ---
{
    const schema = {
        fields: [
            { name: "size", type: "string", enum: ["1024*1024", "2048*2048"], default: "2048*2048" },
            { name: "output_format", type: "string", enum: ["jpeg", "png"], default: "jpeg" },
            { name: "thinking", type: "boolean", default: true },
        ],
    };
    const node = makeParamNode();
    buildParamWidgets(node, schema);
    check(
        "a freshly selected model writes its schema defaults, matching what the widgets show",
        JSON.stringify(readParams(node)) === JSON.stringify({ size: "2048*2048", output_format: "jpeg", thinking: true }),
        JSON.stringify(readParams(node))
    );
}

// --- buildParamWidgets: width/height from an older workflow are carried into `size` ---
{
    const schema = {
        fields: [{ name: "size", type: "string", enum: ["1024*1024", "1328*1776", "2048*2048"], default: "2048*2048" }],
    };
    const node = makeParamNode();
    widgetOf(node, "width").value = 1328;
    widgetOf(node, "height").value = 1776;
    buildParamWidgets(node, schema);
    check(
        "dimensions from a pre-schema workflow become the size value instead of the model default",
        widgetOf(node, "param_size")?.value === "1328*1776",
        widgetOf(node, "param_size")?.value
    );
    check(
        "the carried-over size is what actually gets sent",
        readParams(node).size === "1328*1776",
        JSON.stringify(readParams(node))
    );

    // A size the model does not offer must not be invented; fall back to the
    // schema default rather than sending something that would be rejected.
    const oddNode = makeParamNode();
    widgetOf(oddNode, "width").value = 999;
    widgetOf(oddNode, "height").value = 333;
    buildParamWidgets(oddNode, schema);
    check(
        "dimensions the model does not offer fall back to the schema default",
        widgetOf(oddNode, "param_size")?.value === "2048*2048",
        widgetOf(oddNode, "param_size")?.value
    );

    // Without dimensions there is nothing to carry over.
    const plainNode = makeParamNode();
    buildParamWidgets(plainNode, schema);
    check(
        "with width and height unset the schema default applies",
        widgetOf(plainNode, "param_size")?.value === "2048*2048",
        widgetOf(plainNode, "param_size")?.value
    );

    // A value the user already chose still outranks anything derived.
    const storedNode = makeParamNode({ paramsJsonValue: JSON.stringify({ size: "1024*1024" }) });
    widgetOf(storedNode, "width").value = 1328;
    widgetOf(storedNode, "height").value = 1776;
    buildParamWidgets(storedNode, schema);
    check(
        "a stored size still wins over dimensions carried from width/height",
        widgetOf(storedNode, "param_size")?.value === "1024*1024",
        widgetOf(storedNode, "param_size")?.value
    );
}

// --- buildParamWidgets: a stored params_json value wins over the schema default ---
{
    const schema = { fields: [{ name: "size", type: "string", enum: ["a", "b"], default: "a" }] };
    const node = makeParamNode({ paramsJsonValue: JSON.stringify({ size: "b" }) });
    buildParamWidgets(node, schema);
    check(
        "a value already present in params_json wins over the schema default (saved-workflow case)",
        widgetOf(node, "param_size")?.value === "b",
        widgetOf(node, "param_size")?.value
    );
    check(
        "a value restored from a saved workflow stays written to params_json",
        readParams(node).size === "b",
        JSON.stringify(readParams(node))
    );
}

// --- buildParamWidgets: switching models removes the old param_* widgets --------
{
    const node = makeParamNode();
    buildParamWidgets(node, { fields: [{ name: "size", type: "string", enum: ["a", "c"], default: "a" }] });
    check("sanity: the first model's param widget exists before switching", widgetOf(node, "param_size") !== undefined);

    // Fix 2 means an untouched default is never written in the first place,
    // so touch the first model's widget explicitly (as a real user choosing
    // a non-default value would) -- otherwise there is nothing for the model
    // switch below to actually drop, and the test would prove nothing.
    const sizeWidget = widgetOf(node, "param_size");
    sizeWidget.value = "c";
    sizeWidget.callback();
    check("sanity: the explicitly chosen value is written before switching", readParams(node).size === "c");

    buildParamWidgets(node, { fields: [{ name: "duration", type: "integer", default: 5 }] });
    check("selecting a different model removes the previous model's param_* widgets", widgetOf(node, "param_size") === undefined);
    check("selecting a different model adds the new model's param_* widgets", widgetOf(node, "param_duration") !== undefined);
    check(
        "params_json after switching models drops the old model's value and carries the new model's default",
        JSON.stringify(readParams(node)) === JSON.stringify({ duration: 5 }),
        JSON.stringify(readParams(node))
    );

    // Changing the new model's widget replaces the default it started from.
    const durationWidget = widgetOf(node, "param_duration");
    durationWidget.value = 9;
    durationWidget.callback();
    check(
        "touching the new model's widget writes it to params_json",
        readParams(node).duration === 9,
        JSON.stringify(readParams(node))
    );
}

// --- buildParamWidgets: width/height visibility follows the size field ----------
{
    const node = makeParamNode();
    buildParamWidgets(node, { fields: [{ name: "size", type: "string", enum: ["a"], default: "a" }] });
    check("width is hidden when the schema declares a size field", widgetOf(node, "width")?.hidden === true);
    check("height is hidden when the schema declares a size field", widgetOf(node, "height")?.hidden === true);

    // Losing the size field does not bring them back: a known schema without
    // any dimension field drops width/height from the request entirely.
    buildParamWidgets(node, { fields: [{ name: "duration", type: "integer", default: 5 }] });
    check("width stays hidden for another known schema with no dimension field",
          widgetOf(node, "width")?.hidden === true);
    check("height stays hidden for another known schema with no dimension field",
          widgetOf(node, "height")?.hidden === true);

    // Only an unknown model brings them back, because only then does the node
    // fall back to sending width/height itself.
    buildParamWidgets(node, { fields: [] });
    check("width is visible again for an uncatalogued model",
          widgetOf(node, "width")?.hidden === false);
    check("height is visible again for an uncatalogued model",
          widgetOf(node, "height")?.hidden === false);
}

// --- Fix 3: seed/num_images are hidden once the schema declares its own
// field of the same name, the same mechanism duration already used -- see
// the Python-side precedence tests in test_atlascloud_api.py for why the
// *fixed* widget still wins when it does carry a real value despite being
// hidden (e.g. left over from a previous, non-declaring model). -------------
{
    const node = makeParamNode();
    check("sanity: LEGACY_WIDGET_NAMES includes seed and num_images",
          LEGACY_WIDGET_NAMES.includes("seed") && LEGACY_WIDGET_NAMES.includes("num_images"),
          JSON.stringify(LEGACY_WIDGET_NAMES));

    buildParamWidgets(node, { fields: [{ name: "seed", type: "integer", default: -1 }] });
    check("seed is hidden once the schema declares a seed field",
          widgetOf(node, "seed")?.hidden === true);
    check("num_images stays visible when the schema does not declare it",
          widgetOf(node, "num_images")?.hidden === false);

    buildParamWidgets(node, { fields: [{ name: "num_images", type: "integer", default: 1 }] });
    check("num_images is hidden once the schema declares a num_images field",
          widgetOf(node, "num_images")?.hidden === true);
    check("seed is visible again once the schema no longer declares it",
          widgetOf(node, "seed")?.hidden === false);

    // The catalogue also calls the same setting `n` (wan-2.7, gpt-image-1) or
    // `max_images` (Seedream /sequential). Python routes the num_images widget
    // to whichever the model declares, so its param_* widget supersedes the
    // legacy one under those names too.
    buildParamWidgets(node, { fields: [{ name: "n", type: "integer", default: 1 }] });
    check("num_images is hidden once the schema declares an n field",
          widgetOf(node, "num_images")?.hidden === true);

    buildParamWidgets(node, { fields: [{ name: "max_images", type: "integer", default: 1 }] });
    check("num_images is hidden once the schema declares a max_images field",
          widgetOf(node, "num_images")?.hidden === true);

    buildParamWidgets(node, { fields: [{ name: "size", type: "string", default: "" }] });
    check("num_images is visible again for a model declaring no count field at all",
          widgetOf(node, "num_images")?.hidden === false);
}

// --- buildParamWidgets: any legacy widget is hidden once the schema declares
// a field of the same name (not just width/height/size), so a value set on
// the legacy widget can never silently lose to params_json without the user
// seeing why -----------------------------------------------------------------
{
    const node = makeParamNode();
    buildParamWidgets(node, { fields: [{ name: "duration", type: "integer", default: 5 }] });
    check("duration is hidden once the schema declares a duration field",
          widgetOf(node, "duration")?.hidden === true);
    check("aspect_ratio stays visible when the schema does not declare it",
          widgetOf(node, "aspect_ratio")?.hidden === false);
    check("resolution stays visible when the schema does not declare it",
          widgetOf(node, "resolution")?.hidden === false);
    // width/height have nowhere to go unless the model declares `size` (they
    // are folded into it) or width/height themselves (which then appear as
    // their own param_ widgets). A known schema declaring none of the three
    // -- 754 of WaveSpeed's 953 models, everything sized by aspect_ratio or
    // resolution alone -- drops the values entirely, so the pair is hidden
    // rather than left on screen offering a resolution that never applies.
    check("width/height are hidden for a known schema that declares no dimension field",
          widgetOf(node, "width")?.hidden === true && widgetOf(node, "height")?.hidden === true,
          `width=${widgetOf(node, "width")?.hidden} height=${widgetOf(node, "height")?.hidden}`);
    buildParamWidgets(node, { fields: [{ name: "size", type: "string", default: "1024*1024" }] });
    check("width/height stay hidden for a model whose size string they feed",
          widgetOf(node, "width")?.hidden === true && widgetOf(node, "height")?.hidden === true);
    buildParamWidgets(node, {
        fields: [
            { name: "width", type: "integer", default: 1024 },
            { name: "height", type: "integer", default: 1024 },
        ],
    });
    check("width/height are hidden in favour of their own param_ widgets when declared",
          widgetOf(node, "width")?.hidden === true && widgetOf(node, "height")?.hidden === true);
    buildParamWidgets(node, { fields: [{ name: "duration", type: "integer", default: 5 }] });

    // The empty shape schema_for_model returns for an uncatalogued model (or
    // a failed lookup) must leave every legacy widget visible -- the node
    // then falls back to sending its own conventional fields, so those
    // widgets are the only way to pass the value at all. This is what keeps
    // the width/height rule above from hiding them on a model nobody has
    // indexed yet.
    buildParamWidgets(node, { fields: [] });
    check("a schema declaring none of the legacy fields leaves all of them visible",
          LEGACY_WIDGET_NAMES.every((name) => widgetOf(node, name)?.hidden === false),
          LEGACY_WIDGET_NAMES.map((name) => `${name}=${widgetOf(node, name)?.hidden}`).join(" "));

    // Switching back to a model whose schema does declare duration hides it
    // again, then switching to one that doesn't restores it -- the same
    // reset-on-every-schema-application discipline as width/height above.
    buildParamWidgets(node, { fields: [{ name: "duration", type: "integer", default: 5 }] });
    check("sanity: duration is hidden again for a model that declares it",
          widgetOf(node, "duration")?.hidden === true);
    buildParamWidgets(node, { fields: [{ name: "aspect_ratio", type: "string", default: "16:9" }] });
    check("switching to a model that does not declare duration restores it",
          widgetOf(node, "duration")?.hidden === false);
    check("switching models hides aspect_ratio for the model that now declares it",
          widgetOf(node, "aspect_ratio")?.hidden === true);
}

// --- fetchSchema: cache hygiene, mirroring fetchModels --------------------------
setResponse("/mdpack/atlas/schema?model=vendor%2Fcached-model", { fields: [], image_fields: [] });
{
    const callsBefore = fetchCalls.length;
    const first = await fetchSchema("vendor/cached-model");
    const callsAfterFirst = fetchCalls.length;
    const second = await fetchSchema("vendor/cached-model");
    check("fetchSchema issues one network request per model id", callsAfterFirst === callsBefore + 1, `before=${callsBefore} after=${callsAfterFirst}`);
    check("a second fetchSchema call for the same model id is a cache hit", fetchCalls.length === callsAfterFirst);
    check("the cached call returns the same schema", JSON.stringify(first) === JSON.stringify(second));
}

setResponse("/mdpack/atlas/schema?model=vendor%2Fflaky-model", new Error("network down"));
{
    const callsBeforeFailure = fetchCalls.length;
    const failed = await fetchSchema("vendor/flaky-model");
    check("a failed schema fetch resolves to null instead of throwing", failed === null, JSON.stringify(failed));
    setResponse("/mdpack/atlas/schema?model=vendor%2Fflaky-model", { fields: [{ name: "duration", type: "integer" }], image_fields: [] });
    const retried = await fetchSchema("vendor/flaky-model");
    check(
        "a retry after a failed schema fetch issues a fresh network request instead of replaying the failure",
        fetchCalls.length === callsBeforeFailure + 2,
        `before=${callsBeforeFailure} after=${fetchCalls.length}`
    );
    check("the retried schema fetch returns the newly available schema", Array.isArray(retried?.fields) && retried.fields.length === 1);
}

// --- applySchema: a failed schema fetch never destroys existing params_json -----
setResponse("/mdpack/atlas/schema?model=vendor%2Ffailing-model", new Error("route unreachable"));
{
    const node = makeParamNode({ modelValue: "vendor/failing-model", paramsJsonValue: JSON.stringify({ size: "kept" }) });
    await applySchema(node);
    check(
        "a failed schema fetch leaves an existing params_json value untouched",
        widgetOf(node, "params_json").value === JSON.stringify({ size: "kept" }),
        widgetOf(node, "params_json").value
    );
    check(
        "a failed schema fetch adds no param_* widgets",
        (node.widgets || []).filter((widget) => widget.name.startsWith(PARAM_PREFIX)).length === 0
    );
}

// --- applySchema: stale responses cannot build widgets after a newer model was picked ---
{
    const deferredSchemaA = createDeferred();
    const deferredSchemaB = createDeferred();
    setResponse("/mdpack/atlas/schema?model=vendor%2Frace-model-a", () => deferredSchemaA.promise);
    setResponse("/mdpack/atlas/schema?model=vendor%2Frace-model-b", () => deferredSchemaB.promise);

    const node = makeParamNode({ modelValue: "vendor/race-model-a" });
    const slowApplyForA = applySchema(node); // starts fetching A's schema, not yet resolved

    // Simulate the user selecting model B before A's schema response arrives.
    widgetOf(node, "model").value = "vendor/race-model-b";
    const fastApplyForB = applySchema(node); // starts fetching B's schema

    // B answers first (fast reply); A answers after (slow, and by now stale).
    deferredSchemaB.resolve({ fields: [{ name: "field_b", type: "string" }], image_fields: [] });
    await fastApplyForB;
    deferredSchemaA.resolve({ fields: [{ name: "field_a", type: "string" }], image_fields: [] });
    await slowApplyForA;

    check(
        "a stale, later-resolving schema response does not build widgets after a newer model was applied",
        widgetOf(node, "param_field_b") !== undefined && widgetOf(node, "param_field_a") === undefined,
        `hasFieldA=${widgetOf(node, "param_field_a") !== undefined} hasFieldB=${widgetOf(node, "param_field_b") !== undefined}`
    );
}

// --- applySchema: no-op on a node without a params_json widget (both current
// node types have one; this covers the guard generically, for any future
// MODEL_NODE_TYPES addition that doesn't) ---
setResponse("/mdpack/atlas/schema?model=vendor%2Fvideo-model", { fields: [{ name: "duration", type: "integer", default: 5 }], image_fields: [] });
{
    const node = makeParamNode({ modelValue: "vendor/video-model" });
    node.widgets = node.widgets.filter((widget) => widget.name !== "params_json");
    const callsBefore = fetchCalls.length;
    await applySchema(node);
    check(
        "applySchema never fetches when the node has no params_json widget",
        fetchCalls.length === callsBefore,
        `before=${callsBefore} after=${fetchCalls.length}`
    );
    check(
        "applySchema adds no param_* widgets when the node has no params_json widget",
        (node.widgets || []).filter((widget) => widget.name.startsWith(PARAM_PREFIX)).length === 0
    );
}

// =================================================================================
// Task 9: dynamic image inputs
// =================================================================================

// A node fixture with a real `inputs` array (the ten image slots) alongside
// the widgets applyImageSlots/refreshImageSlots read and write.
function makeImageNode({ modelValue = "vendor/model-a", paramsJsonValue = "{}", connectedSlots = [] } = {}) {
    const node = {
        widgets: [
            { name: "model", type: "combo", value: modelValue, callback: undefined, options: { values: [modelValue] } },
            { name: "params_json", type: "string", value: paramsJsonValue },
            { name: "input_map", type: "string", value: "" },
        ],
        inputs: IMAGE_SLOT_NAMES.map((slotName) => ({
            name: slotName,
            type: "IMAGE",
            link: connectedSlots.includes(slotName) ? 1 : null,
            label: slotName,
            hidden: false,
        })),
        graph: { setDirtyCanvas() {} },
        computeSize: () => [220, 120],
        setSize() {},
        size: [220, 120],
    };
    node.addWidget = (type, name, value, callback, options) => {
        const widget = { type, name, value, callback, options: options ? { ...options } : undefined };
        node.widgets.push(widget);
        return widget;
    };
    return node;
}

function slotOf(node, name) {
    return node.inputs.find((input) => input.name === name);
}

// --- applyImageSlots: a single list field --------------------------------------
{
    const node = makeImageNode();
    applyImageSlots(node, { image_fields: [{ name: "images", is_list: true, max_items: 10 }] });
    const labels = IMAGE_SLOT_NAMES.map((name) => slotOf(node, name).label);
    check(
        "a list field with max_items 10 labels all ten slots images 1..images 10",
        JSON.stringify(labels) === JSON.stringify(Array.from({ length: 10 }, (_, i) => `images ${i + 1}`)),
        JSON.stringify(labels)
    );
    check(
        "all ten slots are visible for a full ten-item list field",
        IMAGE_SLOT_NAMES.every((name) => slotOf(node, name).hidden === false)
    );
    check(
        "a single list field clears input_map, letting Python's default schema order handle it",
        widgetOf(node, "input_map").value === "",
        widgetOf(node, "input_map").value
    );
}

// --- applyImageSlots: a list field with a smaller max_items ---------------------
{
    const node = makeImageNode();
    applyImageSlots(node, { image_fields: [{ name: "images", is_list: true, max_items: 3 }] });
    check(
        "a list field with max_items 3 shows only three slots, labelled images 1..images 3",
        ["images 1", "images 2", "images 3"].every((label, i) => slotOf(node, IMAGE_SLOT_NAMES[i]).label === label)
    );
    check(
        "slots beyond max_items are hidden when not connected",
        IMAGE_SLOT_NAMES.slice(3).every((name) => slotOf(node, name).hidden === true)
    );
}

// --- applyImageSlots: two singular fields ----------------------------------------
{
    const node = makeImageNode({ connectedSlots: ["image", "image_2"] });
    applyImageSlots(node, {
        image_fields: [
            { name: "image", is_list: false, max_items: 1 },
            { name: "last_image", is_list: false, max_items: 1 },
        ],
    });
    check(
        "two singular fields show exactly two visible slots and hide the rest",
        slotOf(node, "image").hidden === false &&
            slotOf(node, "image_2").hidden === false &&
            IMAGE_SLOT_NAMES.slice(2).every((name) => slotOf(node, name).hidden === true)
    );
    check(
        "the two visible slots are labelled with the schema's own field names",
        slotOf(node, "image").label === "image" && slotOf(node, "image_2").label === "last_image"
    );
    check(
        "input_map is written as field-name to connected-index when both slots are connected",
        widgetOf(node, "input_map").value === JSON.stringify({ image: 0, last_image: 1 }),
        widgetOf(node, "input_map").value
    );
}

// --- applyImageSlots: no media fields --------------------------------------------
{
    const node = makeImageNode();
    applyImageSlots(node, { image_fields: [] });
    check(
        "an empty image_fields list hides every image slot",
        IMAGE_SLOT_NAMES.every((name) => slotOf(node, name).hidden === true)
    );
    check("an empty image_fields list clears input_map", widgetOf(node, "input_map").value === "");
}

// --- Fix 4: video/audio-kind media fields are never offered as image slots ------
{
    const node = makeImageNode();
    applyImageSlots(node, {
        image_fields: [{ name: "video", is_list: false, max_items: 1, kind: "video" }],
    });
    check(
        "a schema whose only media field is video-kind hides every image slot",
        IMAGE_SLOT_NAMES.every((name) => slotOf(node, name).hidden === true)
    );
}

// --- Fix 5: a list image field gets the list treatment even with a
// video/audio field declared beside it (8 live models mix these, e.g.
// wan-2.7/reference-to-video: images (list, max 10) + audio) ---------------------
{
    const node = makeImageNode();
    applyImageSlots(node, {
        image_fields: [
            { name: "images", is_list: true, max_items: 10, kind: "image" },
            { name: "audio", is_list: false, max_items: 1, kind: "audio" },
        ],
    });
    const labels = IMAGE_SLOT_NAMES.map((name) => slotOf(node, name).label);
    check(
        "the image-kind list field still gets all ten slots, labelled images 1..images 10, "
            + "despite the audio field sitting beside it in the schema",
        JSON.stringify(labels) === JSON.stringify(Array.from({ length: 10 }, (_, i) => `images ${i + 1}`)),
        JSON.stringify(labels)
    );
    check(
        "the list field clears input_map (Python's schema-order default handles it)",
        widgetOf(node, "input_map").value === "",
        widgetOf(node, "input_map").value
    );
}

// A second real mixed shape from the review: a singular video field plus a
// smaller image list field (kling-video-o3-pro/reference-to-video: video +
// images, max 7).
{
    const node = makeImageNode();
    applyImageSlots(node, {
        image_fields: [
            { name: "video", is_list: false, max_items: 1, kind: "video" },
            { name: "images", is_list: true, max_items: 7, kind: "image" },
        ],
    });
    const labels = IMAGE_SLOT_NAMES.slice(0, 7).map((name) => slotOf(node, name).label);
    check(
        "a video field declared before the image list field still doesn't stop the "
            + "list field from getting all seven of its own slots",
        JSON.stringify(labels) === JSON.stringify(Array.from({ length: 7 }, (_, i) => `images ${i + 1}`)),
        JSON.stringify(labels)
    );
    check(
        "no slots beyond the list field's max_items are used",
        IMAGE_SLOT_NAMES.slice(7).every((name) => slotOf(node, name).hidden === true)
    );
}

// --- applyImageSlots: a connected-but-unused slot is never hidden ---------------
{
    const node = makeImageNode({ connectedSlots: ["image", "image_2", "image_6"] });
    applyImageSlots(node, {
        image_fields: [
            { name: "image", is_list: false, max_items: 1 },
            { name: "last_image", is_list: false, max_items: 1 },
        ],
    });
    const unusedSlot = slotOf(node, "image_6");
    check("a slot with a link stays visible even when the selected model doesn't use it", unusedSlot.hidden === false);
    check(
        "a connected-but-unused slot is relabelled to say the model ignores it",
        unusedSlot.label.includes("unused by this model"),
        unusedSlot.label
    );
}

// --- applyImageSlots: switching models repeatedly leaves no leftover labels -----
{
    const fiveFieldSchema = {
        image_fields: [
            { name: "a", is_list: false, max_items: 1 },
            { name: "b", is_list: false, max_items: 1 },
            { name: "c", is_list: false, max_items: 1 },
            { name: "d", is_list: false, max_items: 1 },
            { name: "e", is_list: false, max_items: 1 },
        ],
    };
    const twoFieldSchema = {
        image_fields: [
            { name: "image", is_list: false, max_items: 1 },
            { name: "last_image", is_list: false, max_items: 1 },
        ],
    };
    const node = makeImageNode();
    applyImageSlots(node, fiveFieldSchema);
    check("sanity: the five-field schema labels slot 5 (image_5) as e", slotOf(node, "image_5").label === "e");

    applyImageSlots(node, twoFieldSchema);
    check(
        "switching to a two-field schema hides and relabels the slots the wider schema had used",
        ["image_3", "image_4", "image_5"].every((name) => slotOf(node, name).hidden === true && slotOf(node, name).label === name),
        JSON.stringify(["image_3", "image_4", "image_5"].map((name) => slotOf(node, name).label))
    );

    applyImageSlots(node, fiveFieldSchema);
    check(
        "switching back to the five-field schema restores its own labels with none left over from the two-field schema",
        slotOf(node, "image").label === "a" &&
            slotOf(node, "image_2").label === "b" &&
            slotOf(node, "image_3").label === "c" &&
            slotOf(node, "image_4").label === "d" &&
            slotOf(node, "image_5").label === "e",
        JSON.stringify(IMAGE_SLOT_NAMES.slice(0, 5).map((name) => slotOf(node, name).label))
    );
}

// --- computeSingularInputMap (via applyImageSlots): a gap before a used slot ----
// The index semantics check the task called out explicitly: atlascloud_api.py's
// collect_reference_images skips unconnected slots when building the reference
// list, so a slot index in input_map must count *connected* slots, not declared
// field position. image_2 is deliberately left unconnected here while image and
// image_3 are connected -- if input_map instead used the naive slot position
// (first=0, third=2), assign_media would either drop the third field's image
// (index out of range) or, in other field counts, hand it to the wrong field.
{
    const node = makeImageNode({ connectedSlots: ["image", "image_3"] });
    applyImageSlots(node, {
        image_fields: [
            { name: "first", is_list: false, max_items: 1 },
            { name: "second", is_list: false, max_items: 1 },
            { name: "third", is_list: false, max_items: 1 },
        ],
    });
    check(
        "input_map indices count connected slots, not declared slot position, so a gap before a used slot doesn't shift its index",
        widgetOf(node, "input_map").value === JSON.stringify({ first: 0, third: 1 }),
        widgetOf(node, "input_map").value
    );
}

// --- computeSingularInputMap: only the second of two singular slots connected --
{
    const node = makeImageNode({ connectedSlots: ["image_2"] });
    applyImageSlots(node, {
        image_fields: [
            { name: "image", is_list: false, max_items: 1 },
            { name: "last_image", is_list: false, max_items: 1 },
        ],
    });
    check(
        "connecting only the second of two singular slots maps it to index 0, matching Python's single-element compacted reference list",
        widgetOf(node, "input_map").value === JSON.stringify({ last_image: 0 }),
        widgetOf(node, "input_map").value
    );
}

// --- refreshImageSlots: a connection made after the model was selected updates input_map ---
{
    const node = makeImageNode({ connectedSlots: ["image"] });
    applyImageSlots(node, {
        image_fields: [
            { name: "image", is_list: false, max_items: 1 },
            { name: "last_image", is_list: false, max_items: 1 },
        ],
    });
    check(
        "sanity: only image is connected right after model selection",
        widgetOf(node, "input_map").value === JSON.stringify({ image: 0 }),
        widgetOf(node, "input_map").value
    );

    // Simulate the user connecting an image to the last_image slot afterwards,
    // without reselecting the model -- this is what onConnectionsChange reacts to.
    slotOf(node, "image_2").link = 1;
    refreshImageSlots(node);
    check(
        "connecting a second slot after the model was selected updates input_map without a fresh schema fetch",
        widgetOf(node, "input_map").value === JSON.stringify({ image: 0, last_image: 1 }),
        widgetOf(node, "input_map").value
    );
}

// --- applySchema: stale schema responses cannot rewrite image slots either ------
// Mirrors the existing "applySchema: stale responses cannot build widgets after a
// newer model was picked" test above, using the same schemaRequestGenerations
// guard that already protects buildParamWidgets -- applyImageSlots is called from
// the same guarded block, so no separate guard was added for it.
{
    const deferredSchemaA = createDeferred();
    const deferredSchemaB = createDeferred();
    setResponse("/mdpack/atlas/schema?model=vendor%2Fimage-race-a", () => deferredSchemaA.promise);
    setResponse("/mdpack/atlas/schema?model=vendor%2Fimage-race-b", () => deferredSchemaB.promise);

    const node = makeImageNode({ modelValue: "vendor/image-race-a" });
    const slowApplyForA = applySchema(node); // starts fetching A's schema, not yet resolved

    // Simulate the user selecting model B before A's schema response arrives.
    widgetOf(node, "model").value = "vendor/image-race-b";
    const fastApplyForB = applySchema(node); // starts fetching B's schema

    // B answers first (fast reply); A answers after (slow, and by now stale).
    deferredSchemaB.resolve({ fields: [], image_fields: [{ name: "images", is_list: true, max_items: 2 }] });
    await fastApplyForB;
    deferredSchemaA.resolve({ fields: [], image_fields: [{ name: "images", is_list: true, max_items: 10 }] });
    await slowApplyForA;

    check(
        "a stale, later-resolving schema response does not rewrite the image slot layout after a newer model was applied",
        slotOf(node, "image_2").hidden === false && slotOf(node, "image_3").hidden === true,
        `image_2.hidden=${slotOf(node, "image_2").hidden} image_3.hidden=${slotOf(node, "image_3").hidden}`
    );
}

// --- provider selection -------------------------------------------------------
// The provider widget decides which catalogue the dropdown shows. Atlas Cloud
// URLs must stay byte-identical to what they were before the widget existed
// (the route defaults to it), while WaveSpeed names itself in the query and
// sends the key in a header rather than the query string.
{
    check("the default provider is Atlas Cloud", DEFAULT_PROVIDER === "atlascloud");

    const node = {
        widgets: [
            { name: "api_key", value: "ws-secret" },
            { name: "provider", value: "wavespeed" },
        ],
    };
    check("providerOf reads the provider widget", providerOf(node) === "wavespeed");
    check("apiKeyOf reads the api_key widget", apiKeyOf(node) === "ws-secret");
    check("a node without the widgets falls back to the default provider and no key",
        providerOf({ widgets: [] }) === DEFAULT_PROVIDER && apiKeyOf({ widgets: [] }) === "");

    const wavespeedUrl = "/mdpack/atlas/models?output=3D&input=Video&provider=wavespeed";
    setResponse(wavespeedUrl, { models: [{ model: "wavespeed-ai/x", display_name: "X" }] });
    const wavespeedModels = await fetchModels("3D", "Video", "wavespeed", "ws-secret");
    check("fetchModels names a non-default provider in the query",
        fetchCalls.at(-1) === wavespeedUrl, fetchCalls.at(-1));
    check("the API key travels in a header, never in the URL",
        fetchOptions.at(-1)?.headers?.[API_KEY_HEADER] === "ws-secret" &&
            !fetchCalls.at(-1).includes("ws-secret"),
        JSON.stringify(fetchOptions.at(-1)));
    check("the WaveSpeed catalogue comes back normally", wavespeedModels.length === 1);

    const atlasUrl = "/mdpack/atlas/models?output=3D&input=Video";
    setResponse(atlasUrl, { models: [{ model: "atlas/x", display_name: "Atlas X" }] });
    const atlasModels = await fetchModels("3D", "Video", "atlascloud", "");
    check("the default provider leaves the Atlas Cloud URL unchanged",
        fetchCalls.at(-1) === atlasUrl, fetchCalls.at(-1));
    check("no header is sent when there is no key", fetchOptions.at(-1) === undefined);
    check("the two providers are cached separately, not served from one entry",
        atlasModels[0].model === "atlas/x" && wavespeedModels[0].model === "wavespeed-ai/x");

    // Same id at both providers: the schema cache must not hand one
    // provider's schema to the other.
    const sharedId = "vendor/shared-id";
    const atlasSchemaUrl = `/mdpack/atlas/schema?model=${encodeURIComponent(sharedId)}`;
    const wavespeedSchemaUrl = `${atlasSchemaUrl}&provider=wavespeed`;
    setResponse(atlasSchemaUrl, { model: sharedId, prompt_field: "atlas_prompt", fields: [], image_fields: [] });
    setResponse(wavespeedSchemaUrl, { model: sharedId, prompt_field: "wavespeed_prompt", fields: [], image_fields: [] });
    const atlasSchema = await fetchSchema(sharedId);
    const wavespeedSchema = await fetchSchema(sharedId, "wavespeed", "ws-secret");
    check("fetchSchema is cached per provider, not per model id alone",
        atlasSchema.prompt_field === "atlas_prompt" &&
            wavespeedSchema.prompt_field === "wavespeed_prompt",
        `${atlasSchema.prompt_field} / ${wavespeedSchema.prompt_field}`);
    check("fetchSchema sends the key as a header too",
        fetchOptions.at(-1)?.headers?.[API_KEY_HEADER] === "ws-secret",
        JSON.stringify(fetchOptions.at(-1)));
}

// --- widgets_values migration --------------------------------------------------
// `provider` is declared first, above the api key it governs. widgets_values
// is positional, so a workflow saved under either earlier layout (provider
// last, or no provider at all) would restore every value one widget too early
// without this repair.
{
    // The static widgets of the image node, in the current declaration order.
    const STATIC_NAMES = [
        "provider", "api_key", "model", "prompt", "width", "height", "num_images",
        "seed", "control_after_generate", "poll_interval", "timeout", "image_field",
        "image_transport", "upload_cache_minutes", "extra_params", "params_json",
        "input_map", "output_type", "input_type", "max_parallel",
    ];

    // INPUT_TYPES()["required"] never lists control_after_generate; the
    // frontend injects that widget itself.
    const REQUIRED_NAMES = STATIC_NAMES.filter((name) => name !== "control_after_generate");

    function makeConfigurableNode() {
        const widgets = STATIC_NAMES.map((name) => ({ name, value: undefined }));
        widgets[0].options = { values: ["atlascloud", "wavespeed"] };
        // The optional input widget sits after the statics and shifts with
        // them; a param_* widget must never be treated as one of them.
        widgets.push({ name: "reference_urls", value: undefined });
        widgets.push({ name: `${PARAM_PREFIX}guidance_scale`, value: 3.5 });
        return { widgets };
    }

    function valuesOf(node) {
        return Object.fromEntries(node.widgets.map((widget) => [widget.name, widget.value]));
    }

    check("declaredWidgetsOf ignores the generated param_ widgets",
        declaredWidgetsOf(makeConfigurableNode()).length === STATIC_NAMES.length + 1);
    check("the required block ends before the optional reference_urls widget",
        requiredWidgetCountOf(makeConfigurableNode(), REQUIRED_NAMES) === STATIC_NAMES.length,
        String(requiredWidgetCountOf(makeConfigurableNode(), REQUIRED_NAMES)));

    // Layout 1: saved before `provider` existed -- one value short.
    const preProvider = [
        "apikey-REDACTED", "bytedance/seedream-v5.0-pro/text-to-image", "a prompt", 1328, 1776,
        1, 3208795585, "randomize", 2, 300, "images", "upload", 120, "", "{}", "", "Image",
        "Any", 4,
    ];
    const preProviderNode = makeConfigurableNode();
    check("a pre-provider workflow is recognised as needing migration",
        migrateWidgetValues(preProviderNode, { widgets_values: preProvider }, REQUIRED_NAMES) === true);
    const preProviderValues = valuesOf(preProviderNode);
    check("the api key lands in api_key, not in provider",
        preProviderValues.api_key === "apikey-REDACTED" &&
            preProviderValues.provider === "atlascloud",
        JSON.stringify(preProviderValues));
    check("every later value is shifted back into its own widget",
        preProviderValues.model === "bytedance/seedream-v5.0-pro/text-to-image" &&
            preProviderValues.prompt === "a prompt" &&
            preProviderValues.width === 1328 &&
            preProviderValues.seed === 3208795585 &&
            preProviderValues.image_field === "images" &&
            preProviderValues.max_parallel === 4,
        JSON.stringify(preProviderValues));
    check("the param_ widget is left untouched by the shift",
        preProviderValues[`${PARAM_PREFIX}guidance_scale`] === 3.5);

    // Layout 2: provider existed, declared last.
    const providerLast = [...preProvider, "wavespeed"];
    const providerLastNode = makeConfigurableNode();
    migrateWidgetValues(providerLastNode, { widgets_values: providerLast }, REQUIRED_NAMES);
    const providerLastValues = valuesOf(providerLastNode);
    check("a provider-last workflow keeps the provider it stored",
        providerLastValues.provider === "wavespeed", JSON.stringify(providerLastValues));
    check("and its other values land in the same widgets as before",
        providerLastValues.api_key === "apikey-REDACTED" &&
            providerLastValues.max_parallel === 4 &&
            providerLastValues.output_type === "Image",
        JSON.stringify(providerLastValues));

    // The optional widgets sit after the statics and shift with them. Getting
    // this wrong is what once left the provider id sitting in reference_urls,
    // from where it was sent to the API as a seventh reference image.
    const withOptional = [...preProvider, "wavespeed", "https://cdn.test/ref.png"];
    const optionalNode = makeConfigurableNode();
    migrateWidgetValues(optionalNode, { widgets_values: withOptional }, REQUIRED_NAMES);
    const optionalValues = valuesOf(optionalNode);
    check("reference_urls keeps its own value across the shift",
        optionalValues.reference_urls === "https://cdn.test/ref.png",
        JSON.stringify(optionalValues));
    check("and never receives the provider id",
        !providerLastNode.widgets.some((widget) =>
            widget.name !== "provider" && widget.value === "wavespeed"),
        JSON.stringify(providerLastValues));

    const preProviderWithOptional = [...preProvider, "https://cdn.test/ref.png"];
    const preProviderOptionalNode = makeConfigurableNode();
    migrateWidgetValues(preProviderOptionalNode, { widgets_values: preProviderWithOptional }, REQUIRED_NAMES);
    check("a pre-provider workflow keeps its reference_urls too",
        valuesOf(preProviderOptionalNode).reference_urls === "https://cdn.test/ref.png",
        JSON.stringify(valuesOf(preProviderOptionalNode)));

    // Layout 3: the current one -- must be left alone.
    const currentLayout = ["wavespeed", ...preProvider];
    const currentNode = makeConfigurableNode();
    check("a workflow already saved provider-first is not migrated again",
        migrateWidgetValues(currentNode, { widgets_values: currentLayout }, REQUIRED_NAMES) === false);
    check("and its widgets are left exactly as ComfyUI restored them",
        currentNode.widgets.every((widget) =>
            widget.value === undefined || widget.name.startsWith(PARAM_PREFIX)),
        JSON.stringify(valuesOf(currentNode)));

    check("a node created fresh (no widgets_values) is not touched",
        migrateWidgetValues(makeConfigurableNode(), {}, REQUIRED_NAMES) === false &&
            migrateWidgetValues(makeConfigurableNode(), undefined, REQUIRED_NAMES) === false &&
            migrateWidgetValues(makeConfigurableNode(), { widgets_values: [] }, REQUIRED_NAMES) === false);
}

console.log();
if (FAILURES.length) {
    console.log(`${FAILURES.length} TEST(S) FAILED: ${FAILURES.join(", ")}`);
    process.exit(1);
}
console.log("ALL TESTS PASSED");
