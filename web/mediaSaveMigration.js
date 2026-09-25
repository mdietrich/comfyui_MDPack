import { app } from "../../scripts/app.js";

const IMAGE_NODE_NAME = "MDPackSaveImageProvenance";
const VIDEO_NODE_NAME = "MDPackSaveVideoProvenance";

const OLD_IMAGE_WIDGET_NAMES = [
    "filename_prefix",
    "format",
    "quality",
    "embed_standalone_ai_xmp",
    "ai_source_type",
    "sign_c2pa",
    "certificate_path",
    "private_key_path",
    "signing_algorithm",
    "c2patool_path",
    "tool_timeout_seconds",
    "embed_workflow",
];

const OLD_VIDEO_WIDGET_NAMES = [
    "filename_prefix",
    "fps",
    "embed_standalone_ai_xmp",
    "ai_source_type",
    "sign_c2pa",
    "certificate_path",
    "private_key_path",
    "signing_algorithm",
    "c2patool_path",
    "tool_timeout_seconds",
    "embed_workflow",
    "ffmpeg_path",
    "ffprobe_path",
    "exiftool_path",
    "input_strategy",
];

function legacyWidgetNamesFor(nodeName) {
    if (nodeName === IMAGE_NODE_NAME) {
        return new Set(["c2patool_path", "tool_timeout_seconds"]);
    }
    if (nodeName === VIDEO_NODE_NAME) return new Set(["c2patool_path"]);
    return new Set();
}

// The Python node still declares these optional names so old API prompts pass
// validation. They are not real controls anymore and must not become new
// graph widgets or add positional values to newly saved workflows.
function removeLegacyApiWidgets(node, nodeName) {
    const legacyNames = legacyWidgetNamesFor(nodeName);
    if (!legacyNames.size || !Array.isArray(node.widgets)) return false;
    let removedAny = false;
    for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (!legacyNames.has(widget.name)) continue;
        widget.onRemove?.();
        widget.element?.remove?.();
        node.widgets.splice(index, 1);
        removedAny = true;
    }
    return removedAny;
}

function looksLikeOldImageLayout(storedValues) {
    return storedValues.length === OLD_IMAGE_WIDGET_NAMES.length &&
        typeof storedValues[9] === "string" &&
        typeof storedValues[10] === "number" &&
        typeof storedValues[11] === "boolean";
}

function looksLikeOldVideoLayout(storedValues) {
    return storedValues.length === OLD_VIDEO_WIDGET_NAMES.length &&
        typeof storedValues[8] === "string" &&
        typeof storedValues[9] === "number" &&
        typeof storedValues[10] === "boolean";
}

// Old graphs stored values positionally. Re-apply them by their old widget
// names after dropping only obsolete c2patool fields; this preserves the video
// timeout and both nodes' embed_workflow booleans. Exact length/type checks
// keep current and future layouts out of this migration.
function migrateMediaSaveWidgetValues(node, nodeName, info) {
    const storedValues = info?.widgets_values;
    if (!Array.isArray(storedValues)) return false;

    let oldWidgetNames;
    if (nodeName === IMAGE_NODE_NAME && looksLikeOldImageLayout(storedValues)) {
        oldWidgetNames = OLD_IMAGE_WIDGET_NAMES;
    } else if (nodeName === VIDEO_NODE_NAME && looksLikeOldVideoLayout(storedValues)) {
        oldWidgetNames = OLD_VIDEO_WIDGET_NAMES;
    } else {
        return false;
    }

    const obsoleteNames = legacyWidgetNamesFor(nodeName);
    const valuesByName = new Map();
    oldWidgetNames.forEach((widgetName, index) => {
        if (!obsoleteNames.has(widgetName)) {
            valuesByName.set(widgetName, storedValues[index]);
        }
    });
    for (const widget of node.widgets || []) {
        if (valuesByName.has(widget.name)) {
            widget.value = valuesByName.get(widget.name);
        }
    }
    return true;
}

app.registerExtension({
    name: "MDPack.MediaSaveMigration",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const nodeName = nodeData.name;
        if (nodeName !== IMAGE_NODE_NAME && nodeName !== VIDEO_NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            removeLegacyApiWidgets(this, nodeName);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            removeLegacyApiWidgets(this, nodeName);
            migrateMediaSaveWidgetValues(this, nodeName, info);
            return result;
        };
    },
});

export {
    IMAGE_NODE_NAME,
    VIDEO_NODE_NAME,
    OLD_IMAGE_WIDGET_NAMES,
    OLD_VIDEO_WIDGET_NAMES,
    removeLegacyApiWidgets,
    migrateMediaSaveWidgetValues,
};
