const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const { webcrypto } = require("node:crypto");

const marker = {
    version: 1, timeline: 1221, widget: "upscaling",
    generation: [1216, 1218, 211, 2003, 2019], final: [1812, 1813], preview: [1617, 1611],
};
const cacheMarker = {
    version: 1, workflow_id: "dcac0945-2963-4788-a355-7dd4de7087dd",
    toggle: 2000, cache: 1605, selector: 2002, generator: 210, display: 211, status: 2003,
};
const outputRoots = [...marker.generation, ...marker.final, ...marker.preview];

function fixture(order = ["mode"], marked = true) {
    const nodes = new Map();
    for (const nodeId of [...outputRoots, 1221, 153, 1811, 2000, 2002, 1605, 210, 212, 233]) {
        nodes.set(nodeId, { id: nodeId, mode: 0, widgets: [], inputs: {}, setDirtyCanvas() {} });
    }
    const widget = { name: "upscaling", value: false };
    nodes.get(1221).widgets = [widget];
    nodes.get(2000).widgets = [{ name: "value", value: true }];
    nodes.get(1605).widgets = [{ name: "value", value: "synthetic cached prompt" }];
    nodes.get(212).widgets = [{ name: "value", value: "synthetic-project" }];
    nodes.get(211).pos = [50000, -50000]; // Deliberately outside any generation group.
    const connections = {
        1216: { samples: ["153", 0] }, 1218: { path: ["1216", 0] },
        153: { conditioning: ["233", 0] }, 233: { text: ["211", 0] },
        211: { text: ["2002", 0] }, 2019: { text: ["211", 0] },
        2003: { text: ["210", 2] },
        2002: { boolean: ["2000", 0], on_true: ["210", 0], on_false: ["1605", 0] },
        1812: { samples: ["1811", 0] }, 1813: { path: ["1812", 0] },
    };
    for (const [nodeId, inputs] of Object.entries(connections)) nodes.get(Number(nodeId)).inputs = inputs;
    for (const nodeId of marker.preview) nodes.get(nodeId).mode = 4;
    const graph = {
        id: "synthetic-actual-workflow-id",
        extra: { h3_prompt_cache: structuredClone(cacheMarker), ...(marked ? { h3_timeline_run_mode: structuredClone(marker) } : {}) },
        getNodeById(nodeId) { return nodes.get(Number(nodeId)); },
        change() { this.changes++; }, changes: 0,
    };
    const extensions = [];
    const listeners = new Map();
    const requests = [];
    const serverApi = {
        addEventListener(name, callback) {
            listeners.set(name, [...(listeners.get(name) ?? []), callback]);
        },
        async queuePrompt(index, prompt, ...arguments_) {
            requests.push({ index, prompt, arguments_ });
            return { prompt_id: String(requests.length) };
        },
    };
    const application = {
        graph, registerExtension(extension) { extensions.push(extension); },
        async graphToPrompt(...arguments_) {
            assert.equal(this, application);
            this.serializedArguments = arguments_;
            this.serializedModes = Object.fromEntries([...nodes].map(([nodeId, node]) => [nodeId, node.mode]));
            const output = {};
            for (const [nodeId, node] of nodes) {
                if (node.mode !== 0) continue;
                output[nodeId] = { class_type: "SyntheticNode", inputs: structuredClone(node.inputs) };
                const valueWidget = node.widgets.find((item) => item.name === "value");
                if (valueWidget) output[nodeId].inputs.value = valueWidget.value;
            }
            return { workflow: {
                id: graph.id, extra: structuredClone(graph.extra),
                nodes: [...nodes.values()].map((node) => ({ id: node.id, widgets_values: node.widgets.map((item) => item.value) })),
            }, output };
        },
        async queuePrompt(index = 0) { return serverApi.queuePrompt(index, await this.graphToPrompt()); },
    };
    for (const extensionName of order) {
        let filename = path.join(__dirname, extensionName === "mode" ? "h3_timeline_run_mode.js" : "h3_prompt_cache.js");
        if (!fs.existsSync(filename)) filename = path.join(__dirname, "../web", path.basename(filename));
        const code = fs.readFileSync(filename, "utf8").replace(/^import .*;\r?\n/gm, "").replace(/export function /g, "function ");
        new Function("app", "api", "crypto", code)(application, serverApi, webcrypto);
    }
    for (const extension of extensions) extension.setup();
    const configure = () => extensions.forEach((extension) => extension.afterConfigureGraph?.());
    const toggle = (value) => { widget.value = value; return widget.callback?.call(widget, value, "callback-argument"); };
    const emit = (name, detail) => (listeners.get(name) ?? []).forEach((callback) => callback({ detail }));
    return { application, graph, nodes, widget, requests, configure, toggle, emit };
}

function executionAncestors(output) {
    const visited = new Set();
    function visit(nodeId) {
        nodeId = String(nodeId);
        if (visited.has(nodeId)) return;
        assert.ok(output[nodeId], `Dangling execution dependency ${nodeId}`);
        visited.add(nodeId);
        for (const value of Object.values(output[nodeId].inputs)) {
            if (Array.isArray(value) && value.length === 2) visit(value[0]);
        }
    }
    for (const nodeId of outputRoots) if (output[nodeId]) visit(nodeId);
    return visited;
}

for (const order of [["mode", "cache"], ["cache", "mode"]]) {
    test(`normal queue generation/final/return with real cache: ${order.join(" then ")}`, async () => {
        const state = fixture(order);
        state.configure();
        assert.equal(state.requests.length, 0);
        // Queue-time normalization must override manual output-root modes.
        for (const nodeId of marker.generation) state.nodes.get(nodeId).mode = 2;
        for (const nodeId of marker.final) state.nodes.get(nodeId).mode = 0;
        await state.application.queuePrompt(3);
        assert.equal(state.requests[0].index, 3);
        let ancestors = executionAncestors(state.requests[0].prompt.output);
        for (const nodeId of [...marker.generation, 153, 210]) assert.ok(ancestors.has(String(nodeId)));
        assert.ok(!ancestors.has("1811"));
        state.toggle(true);
        assert.equal(state.requests.length, 1);
        for (const nodeId of marker.generation) state.nodes.get(nodeId).mode = 0;
        for (const nodeId of marker.preview) state.nodes.get(nodeId).mode = 0;
        await state.application.queuePrompt();
        ancestors = executionAncestors(state.requests[1].prompt.output);
        assert.ok(ancestors.has("1811"));
        for (const nodeId of [...marker.generation, ...marker.preview, 153, 210]) assert.ok(!ancestors.has(String(nodeId)));
        for (const nodeId of marker.generation) assert.equal(state.application.serializedModes[nodeId], 2);
        state.toggle(false);
        await state.application.queuePrompt();
        ancestors = executionAncestors(state.requests[2].prompt.output);
        assert.ok(ancestors.has("153"));
        assert.ok(!ancestors.has("1811"));
        for (const nodeId of marker.preview) assert.equal(state.nodes.get(nodeId).mode, 2);
        await state.application.graphToPrompt("passthrough");
        assert.deepEqual(state.application.serializedArguments, ["passthrough"]);
        assert.equal(state.requests.length, 3);
    });

    test(`OFF SaveText and cache reload hook: ${order.join(" then ")}`, async () => {
        const state = fixture(order);
        state.configure();
        await state.application.queuePrompt();
        state.emit("executed", { prompt_id: "1", node: "211", output: { text: ["synthetic new prompt"] } });
        state.emit("executed", { prompt_id: "1", node: "2003", output: { text: ["TPS: 12"] } });
        assert.equal(state.nodes.get(1605).widgets[0].value, "synthetic new prompt");
        state.nodes.get(1605).widgets[0].value = "stale serialized cache";
        state.configure();
        assert.equal(state.nodes.get(1605).widgets[0].value, "synthetic new prompt");
        state.nodes.get(2000).widgets[0].value = false;
        await state.application.queuePrompt();
        const output = state.requests[1].prompt.output;
        assert.deepEqual(output[2019].inputs.text, ["211", 0]);
        assert.ok(executionAncestors(output).has("1605"));
        assert.ok(!executionAncestors(output).has("210"));
        assert.ok(!output[210]);
    });
}

test("native callback preserves this, arguments, return and chain; toggles never queue", () => {
    const state = fixture();
    let callbackCount = 0;
    state.widget.callback = function (...arguments_) {
        assert.equal(this, state.widget);
        assert.deepEqual(arguments_, [true, "callback-argument"]);
        callbackCount++;
        state.nodes.get(211).mode = 0;
        return "original-result";
    };
    state.configure();
    const bound = state.widget.callback;
    state.configure();
    assert.equal(state.widget.callback, bound);
    assert.equal(state.toggle(true), "original-result");
    assert.equal(callbackCount, 1);
    assert.equal(state.nodes.get(211).mode, 2);
    assert.equal(state.requests.length, 0);
    const previousChanges = state.graph.changes;
    state.configure();
    assert.equal(state.graph.changes, previousChanges);
});

test("reload binds replacement widget and preserves saved preview state", async () => {
    const state = fixture();
    state.configure();
    assert.equal(state.nodes.get(1617).mode, 4);
    const replacement = { name: "upscaling", value: "upscale" };
    state.nodes.get(1221).widgets = [replacement];
    state.configure();
    assert.equal(typeof replacement.callback, "function");
    assert.equal(state.nodes.get(211).mode, 2);
    replacement.value = "false";
    replacement.callback("false");
    await state.application.graphToPrompt();
    assert.equal(state.nodes.get(211).mode, 0);
    assert.equal(state.nodes.get(1617).mode, 2);
});

test("legacy values and queue without configure normalize before serialization", async () => {
    const state = fixture();
    for (const [value, mode] of [["true", 2], ["upscale", 2], [" TRUE ", 2], ["false", 0], [false, 0]]) {
        state.widget.value = value;
        await state.application.graphToPrompt();
        assert.equal(state.application.serializedModes[211], mode);
    }
});

test("missing configured node/widget fails clearly before any queue or mode changes", async () => {
    for (const missingId of [1221, ...outputRoots]) {
        const state = fixture();
        state.nodes.delete(missingId);
        await assert.rejects(state.application.queuePrompt(), new RegExp(`configured node ${missingId} is missing`));
        assert.equal(state.requests.length, 0);
        assert.equal(state.graph.changes, 0);
    }
    const state = fixture();
    state.nodes.get(1221).widgets = [];
    assert.throws(state.configure, /missing widget upscaling/);
    await assert.rejects(state.application.queuePrompt(), /missing widget upscaling/);
    assert.equal(state.requests.length, 0);
});

test("unmarked workflow remains untouched, including original callback", async () => {
    const state = fixture(["mode"], false);
    const callback = () => "unchanged";
    state.widget.callback = callback;
    const originalModes = [...state.nodes.values()].map((node) => node.mode);
    state.configure();
    state.toggle(true);
    await state.application.queuePrompt();
    assert.deepEqual([...state.nodes.values()].map((node) => node.mode), originalModes);
    assert.equal(state.widget.callback, callback);
    assert.equal(state.graph.changes, 0);
    assert.equal(state.requests.length, 1);
});
