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

app.registerExtension({
    name: "MDPack.KeyValueDropdown",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const mappingWidget = this.widgets?.find((w) => w.name === "mapping");
            const selectionWidget = this.widgets?.find((w) => w.name === "selection");
            if (!mappingWidget || !selectionWidget) return result;

            // Render `selection` as a dropdown whose options always reflect
            // the current mapping text.
            selectionWidget.type = "combo";
            selectionWidget.options = selectionWidget.options || {};
            selectionWidget.options.values = () => parseKeys(mappingWidget.value);

            // Mirror the mapping into a node property so the pairs can also
            // be edited via the properties panel.
            if (this.properties?.mapping === undefined) {
                this.addProperty("mapping", mappingWidget.value, "string");
            }
            const originalCallback = mappingWidget.callback;
            mappingWidget.callback = (value, ...rest) => {
                this.properties.mapping = value;
                return originalCallback?.(value, ...rest);
            };
            const onPropertyChanged = this.onPropertyChanged;
            this.onPropertyChanged = function (name, value) {
                if (name === "mapping" && mappingWidget.value !== value) {
                    mappingWidget.value = value ?? "";
                }
                return onPropertyChanged?.apply(this, arguments);
            };
            return result;
        };

        // After a workflow is loaded, the widget values are authoritative —
        // resync the property so panel and widget agree.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            const mappingWidget = this.widgets?.find((w) => w.name === "mapping");
            if (mappingWidget) this.properties.mapping = mappingWidget.value;
            return result;
        };
    },
});
