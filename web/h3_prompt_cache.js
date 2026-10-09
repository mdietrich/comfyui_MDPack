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

function preparePrompt(prompt) {
    const settings = configuration(prompt?.workflow);
    if (!settings || !prompt.output) return { prompt, settings: null, generating: false };
    const output = { ...prompt.output };
    const generating = Boolean(output[settings.display]) && output[settings.toggle]?.inputs?.value === true;
    if (!generating) {
        const cached = output[settings.cache]?.inputs?.value;
        if (output[settings.display] && (typeof cached !== "string" || !cached.trim())) {
            throw new Error("Prompt generation is OFF and no cached prompt is available. Turn prompt generation ON first.");
        }
        if (output[settings.selector]) {
            output[settings.selector] = { ...output[settings.selector], inputs: {
                ...output[settings.selector].inputs,
                on_true: [String(settings.cache), 0], on_false: [String(settings.cache), 0],
            } };
        }
        delete output[settings.generator];
        if (output[settings.status]) {
            output[settings.status] = { ...output[settings.status], inputs: {
                ...output[settings.status].inputs, text: "Prompt reused; OpenRouter disabled",
            } };
        }
    } else {
        if (!output[settings.generator] || !output[settings.status]) {
            throw new Error("Prompt cache wiring is incomplete. Restore OpenRouter and prompt status nodes.");
        }
        output[settings.generator] = { ...output[settings.generator], inputs: {
            ...output[settings.generator].inputs,
            _h3_prompt_cache_nonce: crypto.getRandomValues(new Uint32Array(4)).join("-"),
        } };
    }
    return { prompt: { ...prompt, output }, settings, generating };
}

function exactText(output) {
    return Array.isArray(output?.text) && output.text.length === 1 && typeof output.text[0] === "string"
        ? output.text[0] : null;
}

export function installPromptCache(application, serverApi) {
    const jobs = new Map();
    const earlyEvents = new Map();
    const latestSuccessByWorkflow = new Map();
    let nextEventOrder = 0;
    let pendingCount = 0;
    function applyCachedSuccess() {
        const settings = configuration({extra: application.graph?.extra});
        if (!settings) return;
        const success = latestSuccessByWorkflow.get(settings.workflow_id);
        const cacheNode = application.graph.getNodeById(settings.cache);
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
                jobs.set(response.prompt_id, { settings: prepared.settings, text: null, stats: null, textOrder: 0, statsOrder: 0 });
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
            if (pendingCount && detail?.prompt_id && (kind !== "executed" || ["211", "2003"].includes(String(detail.node)))) {
                if (earlyEvents.size >= 64 && !earlyEvents.has(detail.prompt_id)) earlyEvents.delete(earlyEvents.keys().next().value);
                const events = earlyEvents.get(detail.prompt_id) ?? [];
                if (events.length < 8) events.push({kind, detail, order});
                earlyEvents.set(detail.prompt_id, events);
            }
            return;
        }
        if (kind !== "executed") { jobs.delete(detail.prompt_id); return; }
        if (String(detail.node) === String(job.settings.display)) { job.text = exactText(detail.output); job.textOrder = order; }
        if (String(detail.node) === String(job.settings.status)) { job.stats = exactText(detail.output); job.statsOrder = order; }
        if (job.text === null || job.stats === null) return;
        const text = job.text;
        jobs.delete(detail.prompt_id);
        if (!text.trim() || ERROR_PREFIX.test(text.trimStart()) || !/^TPS:\s*\d/.test(job.stats)) return;
        const completionOrder = Math.max(job.textOrder, job.statsOrder);
        if ((latestSuccessByWorkflow.get(job.settings.workflow_id)?.order ?? -1) > completionOrder) return;
        latestSuccessByWorkflow.set(job.settings.workflow_id, {text, order: completionOrder});
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
        const toggle = app.graph.getNodeById(settings.toggle)?.widgets?.find((item) => item.name === "value");
        if (toggle) toggle.options = { ...toggle.options, on: "ON", off: "OFF" };
    },
});
