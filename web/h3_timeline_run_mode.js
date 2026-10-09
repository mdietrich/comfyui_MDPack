import { app } from "../../scripts/app.js";

const MARKER = "h3_timeline_run_mode";

function fail(message) {
    throw new Error(`H3 Timeline mode: ${message}`);
}

function isFinal(value) {
    if (typeof value === "boolean") return value;
    if (typeof value === "string") {
        const normalized = value.trim().toLowerCase();
        if (["true", "upscale"].includes(normalized)) return true;
        if (["false", "generate"].includes(normalized)) return false;
    }
    fail("upscaling must be a boolean (or true/false/upscale/generate).");
}

function configuredNodes(graph) {
    const settings = graph?.extra?.[MARKER];
    if (settings === undefined) return null;
    if (settings?.version !== 1 || !settings.timeline || typeof settings.widget !== "string") {
        fail("invalid workflow configuration.");
    }
    const findNode = (nodeId) => graph.getNodeById(nodeId) ?? fail(`configured node ${nodeId} is missing.`);
    const timeline = findNode(settings.timeline);
    const widget = timeline.widgets?.find((item) => item.name === settings.widget);
    if (!widget) fail(`node ${settings.timeline} is missing widget ${settings.widget}.`);
    const roots = {};
    for (const group of ["generation", "final", "preview"]) {
        if (!Array.isArray(settings[group]) || !settings[group].length) fail(`missing ${group} node IDs.`);
        roots[group] = settings[group].map(findNode);
    }
    return { widget, roots };
}

export function installTimelineRunMode(application) {
    const callbacks = new WeakMap();
    function synchronize() {
        const graph = application.graph;
        const configured = configuredNodes(graph);
        if (!configured) return;
        const final = isFinal(configured.widget.value);
        let changed = false;
        function setModes(nodes, mode) {
            for (const node of nodes) {
                if (node.mode === mode) continue;
                node.mode = mode;
                node.setDirtyCanvas?.(true, true);
                changed = true;
            }
        }
        setModes(configured.roots.generation, final ? 2 : 0);
        setModes(configured.roots.final, final ? 0 : 2);
        if (final) setModes(configured.roots.preview, 2);
        if (changed) graph.change?.();
    }

    const originalGraphToPrompt = application.graphToPrompt;
    application.graphToPrompt = async function (...arguments_) {
        // Normalize before serialization, including when the cache wrapper is outside us.
        synchronize();
        return originalGraphToPrompt.apply(this, arguments_);
    };

    function afterConfigureGraph() {
        const graph = application.graph;
        const configured = configuredNodes(graph);
        if (!configured) return;
        const widget = configured.widget;
        if (callbacks.get(widget) !== widget.callback || !callbacks.has(widget)) {
            const originalCallback = widget.callback;
            const callback = function (...arguments_) {
                try {
                    return originalCallback?.apply(this, arguments_);
                } finally {
                    if (application.graph === graph) synchronize();
                }
            };
            callbacks.set(widget, callback);
            widget.callback = callback;
        }
        synchronize();
    }
    return { afterConfigureGraph };
}

let controller;
app.registerExtension({
    name: "H3.TimelineRunMode.V1",
    setup() { controller = installTimelineRunMode(app); },
    afterConfigureGraph() { controller?.afterConfigureGraph(); },
});
