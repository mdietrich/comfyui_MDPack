import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MARKER = "h3_prompt_cache";
const WORKFLOW_ID = "dcac0945-2963-4788-a355-7dd4de7087dd";
const ERROR_PREFIX = /^(?:Error: API Key not provided\.|Error processing image_|Error encoding PDF:|API Request Error:|Node Error:)/i;

function configuration(workflow) {
    const settings = workflow?.extra?.[MARKER];
    if (settings?.version !== 1 || settings.workflow_id !== WORKFLOW_ID) return null;
    return settings;
}

// Root graph first, then nodes inside (nested) subgraphs.
function findGraphNode(graph, nodeId) {
    const direct = graph?.getNodeById?.(nodeId);
    if (direct) return direct;
    const matches = new Set();
    const visit = (currentGraph, visitedSubgraphs) => {
        for (const wrapper of currentGraph?.nodes ?? []) {
            const subgraph = wrapper?.isSubgraphNode?.() ? wrapper.subgraph : null;
            if (!subgraph || visitedSubgraphs.has(subgraph)) continue;
            const inner = subgraph.getNodeById(nodeId);
            if (inner) matches.add(inner);
            visit(subgraph, new Set([...visitedSubgraphs, subgraph]));
        }
    };
    visit(graph, new Set());
    if (matches.size > 1) throw new Error(`H3 prompt cache: node ${nodeId} is ambiguous (found in several subgraphs).`);
    return matches.values().next().value ?? null;
}

// Subgraph interior nodes are serialized as "<wrapper>:<node>" execution IDs.
function outputKey(output, nodeId) {
    const exact = String(nodeId);
    if (Object.prototype.hasOwnProperty.call(output, exact)) return exact;
    const keys = Object.keys(output).filter((key) => key.endsWith(`:${exact}`));
    if (keys.length > 1) throw new Error(`H3 prompt cache: node ${exact} is ambiguous (several subgraph instances).`);
    return keys[0] ?? exact;
}

function cacheScope(workflowId, project) {
    if (typeof workflowId !== "string" || !workflowId.trim() || typeof project !== "string" || !project.trim()) return null;
    return JSON.stringify([workflowId, project]);
}

function promptScope(prompt, settings) {
    const projectId = settings.project ?? 212;
    const serializedNodes = [
        ...(prompt.workflow.nodes ?? []),
        ...(prompt.workflow.definitions?.subgraphs ?? []).flatMap((subgraph) => subgraph.nodes ?? []),
    ];
    const projectNode = serializedNodes.find((node) => String(node.id) === String(projectId));
    // Positional widget values are the serialized source; named values may be stale.
    const hasPositionalProject = Object.prototype.hasOwnProperty.call(projectNode?.widgets_values ?? {}, 0);
    const hasSerializedProject = hasPositionalProject || Object.prototype.hasOwnProperty.call(projectNode?.widgets_values_named ?? {}, "value");
    const serializedProject = hasPositionalProject ? projectNode.widgets_values[0] : projectNode?.widgets_values_named?.value;
    if (hasSerializedProject && (typeof serializedProject !== "string" || !serializedProject.trim())) return null;
    const outputProject = prompt.output[outputKey(prompt.output, projectId)]?.inputs?.value;
    if (serializedProject !== undefined && outputProject !== undefined && serializedProject !== outputProject) return null;
    return cacheScope(prompt.workflow.id, hasSerializedProject ? serializedProject : outputProject);
}

function preparePrompt(prompt) {
    const settings = configuration(prompt?.workflow);
    if (!settings || !prompt.output) return { prompt, settings: null, generating: false };
    const output = { ...prompt.output };
    const ids = Object.fromEntries(["toggle", "cache", "selector", "generator", "display", "status"]
        .map((name) => [name, outputKey(output, settings[name])]));
    const generating = Boolean(output[ids.display]) && output[ids.toggle]?.inputs?.value === true;
    if (!generating) {
        const cached = output[ids.cache]?.inputs?.value;
        if (output[ids.display] && (typeof cached !== "string" || !cached.trim())) {
            throw new Error("Prompt generation is OFF and no cached prompt is available. Turn prompt generation ON first.");
        }
        if (output[ids.selector]) {
            output[ids.selector] = { ...output[ids.selector], inputs: {
                ...output[ids.selector].inputs,
                on_true: [ids.cache, 0], on_false: [ids.cache, 0],
            } };
        }
        delete output[ids.generator];
        if (output[ids.status]) {
            output[ids.status] = { ...output[ids.status], inputs: {
                ...output[ids.status].inputs, text: "Prompt reused; OpenRouter disabled",
            } };
        }
    } else {
        if (!output[ids.generator] || !output[ids.status]) {
            throw new Error("Prompt cache wiring is incomplete. Restore OpenRouter and prompt status nodes.");
        }
        output[ids.generator] = { ...output[ids.generator], inputs: {
            ...output[ids.generator].inputs,
            _h3_prompt_cache_nonce: crypto.getRandomValues(new Uint32Array(4)).join("-"),
        } };
    }
    return { prompt: { ...prompt, output }, settings, ids, generating, scope: promptScope(prompt, settings) };
}

function exactText(output) {
    return Array.isArray(output?.text) && output.text.length === 1 && typeof output.text[0] === "string"
        ? output.text[0] : null;
}

export function installPromptCache(application, serverApi) {
    const jobs = new Map();
    const earlyEvents = new Map();
    const latestSuccessByScope = new Map();
    let nextEventOrder = 0;
    let pendingCount = 0;
    function applyCachedSuccess() {
        const settings = configuration({extra: application.graph?.extra});
        if (!settings) return;
        const project = findGraphNode(application.graph, settings.project ?? 212)?.widgets?.find((item) => item.name === "value")?.value;
        const scope = cacheScope(application.graph.id, project);
        if (!scope) return;
        const success = latestSuccessByScope.get(scope);
        const cacheNode = findGraphNode(application.graph, settings.cache);
        const widget = cacheNode?.widgets?.find((item) => item.name === "value");
        if (!success || !widget || widget.value === success.text) return;
        widget.value = success.text;
        if (cacheNode.widgets_values_named) cacheNode.widgets_values_named.value = success.text;
        widget.callback?.(success.text);
        cacheNode.setDirtyCanvas?.(true, true);
        application.graph.change?.();
    }
    const originalGraphToPrompt = application.graphToPrompt;
    application.graphToPrompt = async function (...arguments_) {
        return preparePrompt(await originalGraphToPrompt.apply(this, arguments_)).prompt;
    };
    const originalQueuePrompt = serverApi.queuePrompt;
    serverApi.queuePrompt = async function (index, prompt, ...arguments_) {
        const prepared = preparePrompt(prompt);
        if (prepared.generating) pendingCount++;
        try {
            const response = await originalQueuePrompt.call(this, index, prepared.prompt, ...arguments_);
            if (prepared.generating && response?.prompt_id) {
                jobs.set(response.prompt_id, { ids: prepared.ids, scope: prepared.scope, text: null, stats: null, textOrder: 0, statsOrder: 0 });
                for (const event of earlyEvents.get(response.prompt_id) ?? []) handleEvent(event.kind, event.detail, event.order);
                earlyEvents.delete(response.prompt_id);
            }
            return response;
        } finally {
            if (prepared.generating) pendingCount--;
            if (pendingCount === 0) earlyEvents.clear();
        }
    };
    function handleEvent(kind, detail, order = ++nextEventOrder) {
        const job = jobs.get(detail?.prompt_id);
        if (!job) {
            if (pendingCount && detail?.prompt_id && (kind !== "executed" || ["211", "2003"].includes(String(detail.node).split(":").pop()))) {
                if (earlyEvents.size >= 64 && !earlyEvents.has(detail.prompt_id)) earlyEvents.delete(earlyEvents.keys().next().value);
                const events = earlyEvents.get(detail.prompt_id) ?? [];
                if (events.length < 8) events.push({kind, detail, order});
                earlyEvents.set(detail.prompt_id, events);
            }
            return;
        }
        if (kind !== "executed") { jobs.delete(detail.prompt_id); return; }
        if (String(detail.node) === job.ids.display) { job.text = exactText(detail.output); job.textOrder = order; }
        if (String(detail.node) === job.ids.status) { job.stats = exactText(detail.output); job.statsOrder = order; }
        if (job.text === null || job.stats === null) return;
        const text = job.text;
        jobs.delete(detail.prompt_id);
        if (!text.trim() || ERROR_PREFIX.test(text.trimStart()) || !/^TPS:\s*\d/.test(job.stats)) return;
        if (!job.scope) return;
        const completionOrder = Math.max(job.textOrder, job.statsOrder);
        if ((latestSuccessByScope.get(job.scope)?.order ?? -1) > completionOrder) return;
        latestSuccessByScope.set(job.scope, {text, order: completionOrder});
        applyCachedSuccess();
    }
    serverApi.addEventListener("executed", ({ detail }) => handleEvent("executed", detail));
    for (const eventName of ["execution_error", "execution_interrupted", "execution_success"]) {
        serverApi.addEventListener(eventName, ({ detail }) => handleEvent(eventName, detail));
    }
    return {afterConfigureGraph: applyCachedSuccess};
}

let controller;
app.registerExtension({
    name: "H3.FantasticPromptCache.V1",
    setup() { controller = installPromptCache(app, api); },
    afterConfigureGraph() {
        controller?.afterConfigureGraph();
        const settings = configuration({ extra: app.graph?.extra });
        if (!settings) return;
        const toggle = findGraphNode(app.graph, settings.toggle)?.widgets?.find((item) => item.name === "value");
        if (toggle) toggle.options = { ...toggle.options, on: "ON", off: "OFF" };
    },
});
