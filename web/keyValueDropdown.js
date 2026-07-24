import { app } from "../../scripts/app.js";

const NODE_NAME = "KeyValueDropdown";

function parseKeys(mappingText) {
    return (mappingText || "")
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith("#") && line.includes("="))
        .map((line) => line.split("=")[0].trim())
        .filter((key) => key.length > 0);
}

function getWidgets(node) {
    const mappingWidget = node.widgets?.find((w) => w.name === "mapping");
    const selectionWidget = node.widgets?.find((w) => w.name === "selection");
    return mappingWidget && selectionWidget ? { mappingWidget, selectionWidget } : null;
}

// Show or hide the mapping textarea. The visibility is stored in a node
// property so it survives save/reload; the button widget itself is not
// serialized.
function setMappingVisible(node, visible) {
    const widgets = getWidgets(node);
    if (!widgets) return;
    const wasVisible = widgets.mappingWidget.hidden !== true;
    widgets.mappingWidget.hidden = !visible;
    node.properties.showMapping = visible;
    const toggleWidget = node.widgets?.find((w) => w.name === "mapping_toggle");
    if (toggleWidget) {
        toggleWidget.label = visible ? "Hide mapping ▴" : "Edit mapping ▾";
    }
    // Only resize on an actual visibility change (keeps a user-resized node
    // intact when the state is re-applied after workflow load), and preserve
    // the current width.
    if (wasVisible !== visible) {
        const computedSize = node.computeSize();
        node.setSize([Math.max(node.size[0], computedSize[0]), computedSize[1]]);
    }
    node.graph?.setDirtyCanvas(true, true);
}

// Rebuild the dropdown option list from the mapping text.
function updateOptions(node) {
    const widgets = getWidgets(node);
    if (!widgets) return [];
    const keys = parseKeys(widgets.mappingWidget.value);
    widgets.selectionWidget.options = widgets.selectionWidget.options || {};
    widgets.selectionWidget.options.values = keys;
    return keys;
}

app.registerExtension({
    name: "MDPack.KeyValueDropdown",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const widgets = getWidgets(this);
            if (!widgets) return result;
            const { mappingWidget, selectionWidget } = widgets;
            updateOptions(this);

            // Mirror the mapping into a node property so the pairs can also
            // be edited via the properties panel.
            if (this.properties?.mapping === undefined) {
                this.addProperty("mapping", mappingWidget.value, "string");
            }
            if (this.properties?.showMapping === undefined) {
                this.addProperty("showMapping", false, "boolean");
            }

            // Toggle button sits below the selection dropdown; the mapping
            // textarea is hidden by default and only shown on demand.
            const toggleWidget = this.addWidget("button", "mapping_toggle", null, () => {
                setMappingVisible(this, !this.properties.showMapping);
            });
            toggleWidget.serialize = false;
            if (toggleWidget.options) toggleWidget.options.serialize = false;
            setMappingVisible(this, this.properties.showMapping === true);
            const originalCallback = mappingWidget.callback;
            mappingWidget.callback = (value, ...rest) => {
                this.properties.mapping = value;
                const keys = updateOptions(this);
                if (keys.length && !keys.includes(selectionWidget.value)) {
                    selectionWidget.value = keys[0];
                }
                return originalCallback?.(value, ...rest);
            };
            const onPropertyChanged = this.onPropertyChanged;
            this.onPropertyChanged = function (name, value) {
                if (name === "mapping" && mappingWidget.value !== value) {
                    mappingWidget.value = value ?? "";
                    updateOptions(this);
                }
                return onPropertyChanged?.apply(this, arguments);
            };
            return result;
        };

        // After a workflow is loaded, the widget values are authoritative —
        // resync the property and rebuild the options (never touch the
        // stored selection here).
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            const widgets = getWidgets(this);
            if (widgets) {
                this.properties.mapping = widgets.mappingWidget.value;
                updateOptions(this);
                setMappingVisible(this, this.properties.showMapping === true);
            }
            return result;
        };
    },
});
