const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');
const { webcrypto } = require('node:crypto');

const marker = {version: 1, workflow_id: 'dcac0945-2963-4788-a355-7dd4de7087dd',
    toggle: 2000, cache: 1605, selector: 2002, generator: 210, display: 211, encoder: 233, status: 2003};
const extensionCode = fs.readFileSync(process.argv[2] || `${__dirname}/h3_prompt_cache.js`, 'utf8')
    .replace(/^import .*;\n/gm, '').replace('export function installPromptCache', 'function installPromptCache')
    + '\nglobalThis.install = installPromptCache;';

function environment({enabled = false, cached = '  saved\nexact text\n', scoped = true, final = false, early = false, deferred = false, serializeCacheWidget = false} = {}) {
    const cacheNode = {widgets: [{name: 'value', value: cached}], widgets_values_named: {value: cached}, setDirtyCanvas() {}};
    const projectNode = {widgets: [{name: 'value', value: 'project-a'}]};
    const graph = {id: '9a49e1b9-d6bf-4ab1-b38f-f506f00b9ad8', extra: scoped ? {h3_prompt_cache: {...marker}} : {}, changed: 0,
        getNodeById(nodeId) { return Number(nodeId) === 1605 ? cacheNode : Number(nodeId) === 212 ? projectNode : null; }, change() { this.changed++; }};
    const listeners = new Map();
    const requests = [];
    const queueResolvers = [];
    const input = {workflow: {id: graph.id, extra: structuredClone(graph.extra), nodes: [{id: 212, widgets_values: ['project-a']}]}, output: {
        212: {class_type: 'PrimitiveStringMultiline', inputs: {value: 'project-a'}},
        210: {class_type: 'OpenRouterNode', inputs: {user_message_input: 'unchanged request', system_prompt: 'unchanged system'}},
        211: {class_type: 'ShowText|pysssss', inputs: {text: ['2002', 0]}},
        233: {class_type: 'MiniMaxH3PromptBuilder', inputs: {prompt_text: ['211', 0]}},
        2000: {class_type: 'PrimitiveBoolean', inputs: {value: enabled}},
        1605: {class_type: 'PrimitiveStringMultiline', inputs: {value: cached}},
        2002: {class_type: 'LazySwitch (obvpm)', inputs: {boolean: ['2000', 0], on_true: ['210', 0], on_false: ['1605', 0]}},
        2003: {class_type: 'ShowText|pysssss', inputs: {text: ['210', 2]}},
    }};
    if (final) { delete input.output[211]; delete input.output[2003]; }
    const application = {graph, registerExtension() {}, async graphToPrompt() {
        const prompt = structuredClone(input);
        if (serializeCacheWidget) {
            prompt.output[1605].inputs.value = cacheNode.widgets[0].value;
            prompt.workflow.nodes.push({id: 1605, widgets_values: [cacheNode.widgets[0].value],
                widgets_values_named: structuredClone(cacheNode.widgets_values_named)});
        }
        return prompt;
    }};
    const serverApi = {async queuePrompt(index, prompt) {
        requests.push(prompt);
        const promptId = `job-${requests.length}`;
        if (early) {
            for (const [node, text] of [[211, 'early successful text'], [2003, 'TPS: 12.34, synthetic stats']]) {
                listeners.get('executed')({detail: {prompt_id: promptId, node, output: {text: [text]}}});
            }
            listeners.get('execution_error')({detail: {prompt_id: promptId}});
        }
        if (deferred) return new Promise((resolve) => queueResolvers.push(() => resolve({prompt_id: promptId})));
        return {prompt_id: promptId};
    },
        addEventListener(name, callback) { listeners.set(name, callback); }};
    const browserCrypto = {getRandomValues: (values) => webcrypto.getRandomValues(values)};
    const context = {app: application, api: serverApi, crypto: browserCrypto, console};
    vm.createContext(context); vm.runInContext(extensionCode, context); const controller = context.install(application, serverApi);
    return {application, serverApi, graph, cacheNode, projectNode, requests, input, controller, browserCrypto,
        resolveQueue(index) { queueResolvers[index](); },
        emit(name, detail) { listeners.get(name)?.({detail}); },
        executed(promptId, text, node = 211) { listeners.get('executed')({detail: {prompt_id: promptId, node, output: {text: [text]}}}); },
        success(promptId, text, statsFirst = false) {
            const events = [[211, text], [2003, 'TPS: 12.34, synthetic stats']];
            for (const [node, value] of statsFirst ? events.reverse() : events) {
                listeners.get('executed')({detail: {prompt_id: promptId, node, output: {text: [value]}}});
            }
        },
        async queue() { return serverApi.queuePrompt(0, await application.graphToPrompt()); }};
}

test('OFF removes OpenRouter and routes both lazy branches to exact cached text without mutating input', async () => {
    const state = environment();
    await state.queue();
    const output = state.requests[0].output;
    assert.equal(output[210], undefined);
    assert.deepEqual(Array.from(output[2002].inputs.on_true), ['1605', 0]);
    assert.deepEqual(Array.from(output[2002].inputs.on_false), ['1605', 0]);
    assert.equal(output[1605].inputs.value, '  saved\nexact text\n');
    assert.equal(output[2003].inputs.text, 'Prompt reused; OpenRouter disabled');
    assert.ok(state.input.output[210]);
    state.executed('job-1', 'must not replace OFF cache');
    assert.equal(state.cacheNode.widgets[0].value, '  saved\nexact text\n');
});

test('empty OFF cache fails before any queue request', async () => {
    const state = environment({cached: '\n  '});
    await assert.rejects(state.queue(), /OFF and no cached prompt/);
    assert.equal(state.requests.length, 0);
});

test('ON generates a fresh scalar nonce without changing provider inputs', async () => {
    const state = environment({enabled: true});
    await state.queue(); await state.queue();
    const first = state.requests[0].output[210].inputs;
    const second = state.requests[1].output[210].inputs;
    assert.notEqual(first._h3_prompt_cache_nonce, second._h3_prompt_cache_nonce);
    assert.equal(typeof first._h3_prompt_cache_nonce, 'string');
    assert.equal(first.user_message_input, 'unchanged request');
    assert.equal(first.system_prompt, 'unchanged system');
});

test('ON works on HTTP with getRandomValues available but randomUUID absent', async () => {
    const state = environment({enabled: true});
    assert.equal(state.browserCrypto.randomUUID, undefined);
    assert.equal(typeof state.browserCrypto.getRandomValues, 'function');
    await state.queue(); await state.queue();
    const first = state.requests[0].output[210].inputs;
    const second = state.requests[1].output[210].inputs;
    assert.match(first._h3_prompt_cache_nonce, /^\d+-\d+-\d+-\d+$/);
    assert.notEqual(first._h3_prompt_cache_nonce, second._h3_prompt_cache_nonce);
    assert.equal(first.user_message_input, 'unchanged request');
    assert.equal(first.system_prompt, 'unchanged system');
});

test('successful event updates exact cache and named serialization before later sampler failure', async () => {
    const state = environment({enabled: true});
    await state.queue();
    const successful = '  successful\nnew prompt\n';
    state.success('job-1', successful);
    state.emit('execution_error', {prompt_id: 'job-1'});
    assert.equal(state.cacheNode.widgets[0].value, successful);
    assert.equal(state.cacheNode.widgets_values_named.value, successful);
    assert.equal(state.graph.changed, 1);
    const saved = JSON.parse(JSON.stringify({widgets_values: [state.cacheNode.widgets[0].value]}));
    assert.equal(saved.widgets_values[0], successful);
});

test('known provider errors and empty responses never replace last success', async () => {
    for (const error of ['Error: API Key not provided.', 'Error processing image_1: failure',
        'Error encoding PDF: failure', 'API Request Error: failure', 'Node Error: failure', '', '\n  ']) {
        const state = environment({enabled: true});
        await state.queue(); state.success('job-1', error);
        assert.equal(state.cacheNode.widgets[0].value, '  saved\nexact text\n');
        assert.equal(state.graph.changed, 0);
    }
});

test('unrelated prompt IDs cannot update; inactive workflow success restores on return only', async () => {
    const state = environment({enabled: true});
    await state.queue(); state.success('other-tab-job', 'foreign response');
    assert.equal(state.graph.changed, 0);
    state.application.graph = {...state.graph, extra: {}};
    state.success('job-1', 'response after workflow switch');
    assert.equal(state.graph.changed, 0);
    assert.equal(state.application.graph.changed, 0);
    state.application.graph = state.graph;
    state.controller.afterConfigureGraph();
    assert.equal(state.cacheNode.widgets[0].value, 'response after workflow switch');
    assert.equal(state.graph.changed, 1);
});

test('workflow marker mismatch never updates a different loaded graph', async () => {
    const state = environment({enabled: true});
    await state.queue();
    state.graph.extra = {h3_prompt_cache: {...marker, workflow_id: 'different-workflow'}};
    state.success('job-1', 'stale response');
    state.controller.afterConfigureGraph();
    assert.equal(state.graph.changed, 0);
});

test('last completion wins even when an older submission runs after a queue-front job', async () => {
    const state = environment({enabled: true});
    await state.queue(); await state.queue();
    state.success('job-2', 'newest success'); state.success('job-1', 'older success');
    assert.equal(state.cacheNode.widgets[0].value, 'older success');
});

test('both UI event orders commit, but incomplete or failed statistics do not', async () => {
    for (const statsFirst of [false, true]) {
        const state = environment({enabled: true});
        await state.queue(); state.success('job-1', 'successful response', statsFirst);
        assert.equal(state.cacheNode.widgets[0].value, 'successful response');
    }
    const state = environment({enabled: true});
    await state.queue(); state.executed('job-1', 'response without success stats');
    assert.equal(state.graph.changed, 0);
    state.executed('job-1', 'Stats N/A due to error', 2003);
    assert.equal(state.graph.changed, 0);
});

test('events before queue response are correlated and replayed before terminal error', async () => {
    const state = environment({enabled: true, early: true});
    await state.queue();
    assert.equal(state.cacheNode.widgets[0].value, 'early successful text');
    assert.equal(state.graph.changed, 1);
});

test('buffer replay preserves receipt completion order despite reversed HTTP responses', async () => {
    const state = environment({enabled: true, deferred: true});
    const first = state.queue(); const second = state.queue();
    await new Promise(setImmediate);
    state.success('job-2', 'completed first');
    state.success('job-1', 'completed last');
    state.resolveQueue(0); await first;
    state.resolveQueue(1); await second;
    assert.equal(state.cacheNode.widgets[0].value, 'completed last');
});

test('final mode removes OpenRouter even when prompt toggle is ON and cache empty', async () => {
    const state = environment({enabled: true, final: true, cached: ''});
    await state.queue();
    assert.equal(state.requests[0].output[210], undefined);
    state.executed('job-1', 'must not update final');
    assert.equal(state.graph.changed, 0);
});

test('unmarked original workflow is entirely untouched', async () => {
    const state = environment({enabled: true, scoped: false});
    await state.queue();
    assert.equal(JSON.stringify(state.requests[0]), JSON.stringify(state.input));
    state.executed('job-1', 'original workflow response');
    assert.equal(state.graph.changed, 0);
});

function selectContext(state, workflowId, project, savedText) {
    state.graph.id = workflowId;
    state.projectNode.widgets[0].value = project;
    state.cacheNode.widgets[0].value = savedText;
    state.cacheNode.widgets_values_named.value = savedText;
    state.input.workflow.id = workflowId;
    state.input.workflow.nodes[0].widgets_values[0] = project;
    state.input.output[212].inputs.value = project;
    state.controller.afterConfigureGraph();
}

for (const dimension of ['graph', 'project']) {
    test(`separate ${dimension} contexts preserve saved text, isolate late success, and restore both`, async () => {
        const state = environment({enabled: true});
        const firstId = state.graph.id;
        const secondId = dimension === 'graph' ? '30462f38-142c-4478-9a2b-cc8b9b5bf251' : firstId;
        const secondProject = dimension === 'project' ? 'project-b' : 'project-a';
        await state.queue(); state.success('job-1', 'first success');
        await state.queue();
        selectContext(state, secondId, secondProject, '  second saved\n');
        assert.equal(state.cacheNode.widgets[0].value, '  second saved\n');
        await state.queue(); state.success('job-3', 'second success');
        state.success('job-2', 'first late success');
        assert.equal(state.cacheNode.widgets[0].value, 'second success');
        selectContext(state, firstId, 'project-a', 'first saved');
        assert.equal(state.cacheNode.widgets[0].value, 'first late success');
        selectContext(state, secondId, secondProject, 'second saved');
        assert.equal(state.cacheNode.widgets[0].value, 'second success');
    });
}

test('direct API queue uses serialized origin even when active graph differs and queue response is delayed', async () => {
    const state = environment({enabled: true, deferred: true});
    const firstId = state.graph.id;
    const queuedPrompt = await state.application.graphToPrompt();
    selectContext(state, '30462f38-142c-4478-9a2b-cc8b9b5bf251', 'project-b', 'second saved');
    const queued = state.serverApi.queuePrompt(0, queuedPrompt);
    queuedPrompt.workflow.id = state.graph.id;
    queuedPrompt.workflow.nodes[0].widgets_values[0] = 'project-b';
    queuedPrompt.output[212].inputs.value = 'project-b';
    state.success('job-1', 'origin result');
    state.resolveQueue(0); await queued;
    assert.equal(state.cacheNode.widgets[0].value, 'second saved');
    selectContext(state, firstId, 'project-a', 'first saved');
    assert.equal(state.cacheNode.widgets[0].value, 'origin result');
});

test('missing or ambiguous queued identity never restores a success', async () => {
    for (const invalidate of [
        (prompt) => { delete prompt.workflow.id; },
        (prompt) => { prompt.workflow.id = ''; },
        (prompt) => { prompt.workflow.nodes = []; delete prompt.output[212]; },
        (prompt) => { prompt.workflow.nodes[0].widgets_values[0] = ''; prompt.output[212].inputs.value = ''; },
        (prompt) => { prompt.output[212].inputs.value = 'different-project'; },
    ]) {
        const state = environment({enabled: true});
        invalidate(state.input);
        await state.queue(); state.success('job-1', 'unscoped result');
        state.controller.afterConfigureGraph();
        assert.equal(state.graph.changed, 0);
        assert.equal(state.cacheNode.widgets[0].value, '  saved\nexact text\n');
    }
});

test('missing active graph ID or project widget fails closed, then restores when both return', async () => {
    for (const missing of ['id', 'project']) {
        const state = environment({enabled: true});
        const firstId = state.graph.id;
        await state.queue();
        if (missing === 'id') delete state.graph.id;
        else state.projectNode.widgets[0].value = '';
        state.success('job-1', 'scoped result');
        state.controller.afterConfigureGraph();
        assert.equal(state.graph.changed, 0);
        selectContext(state, firstId, 'project-a', 'saved');
        assert.equal(state.cacheNode.widgets[0].value, 'scoped result');
    }
});

test('serialized positional or named project and API output fallback support explicit project node setting', async () => {
    for (const source of ['positional', 'named', 'output', 'custom']) {
        const state = environment({enabled: true});
        if (source === 'positional') delete state.input.output[212];
        if (source === 'named') {
            state.input.workflow.nodes[0].widgets_values_named = {value: 'project-a'};
            delete state.input.workflow.nodes[0].widgets_values;
            delete state.input.output[212];
        }
        if (source === 'output') state.input.workflow.nodes = [];
        if (source === 'custom') {
            state.input.workflow.extra.h3_prompt_cache.project = 2999;
            state.graph.extra.h3_prompt_cache.project = 2999;
            state.input.workflow.nodes[0].id = 2999;
            state.input.output[2999] = state.input.output[212];
            delete state.input.output[212];
            const originalGetNode = state.graph.getNodeById;
            state.graph.getNodeById = (nodeId) => Number(nodeId) === 2999 ? state.projectNode : originalGetNode(nodeId);
        }
        await state.queue(); state.success('job-1', `result from ${source}`);
        assert.equal(state.cacheNode.widgets[0].value, `result from ${source}`);
    }
});

test('project identity keeps exact whitespace and ignores stale named value when positional value exists', async () => {
    const state = environment({enabled: true});
    state.input.workflow.nodes[0].widgets_values_named = {value: 'stale-project'};
    await state.queue(); state.success('job-1', 'exact project success');
    assert.equal(state.cacheNode.widgets[0].value, 'exact project success');
    selectContext(state, state.graph.id, ' project-a ', 'different saved');
    assert.equal(state.cacheNode.widgets[0].value, 'different saved');
});


test('explicit invalid positional project never falls back to stale named project without API node', async () => {
    for (const invalidProject of [null, undefined, 42, false, {}, [], '', ' \n ']) {
        const state = environment({enabled: true});
        state.input.workflow.nodes[0].widgets_values = [invalidProject];
        state.input.workflow.nodes[0].widgets_values_named = {value: 'project-a'};
        delete state.input.output[212];
        await state.queue();
        state.success('job-1', 'must not cache invalid scope');
        state.controller.afterConfigureGraph();
        assert.equal(state.graph.changed, 0);
        assert.equal(state.cacheNode.widgets[0].value, '  saved\nexact text\n');
        assert.equal(state.cacheNode.widgets_values_named.value, '  saved\nexact text\n');
    }
});

test('success then downstream error then OFF retry serializes and reuses exact updated cache', async () => {
    const state = environment({enabled: true, serializeCacheWidget: true});
    const successfulText = '  generated\nprompt with trailing spaces  \n';
    await state.queue();
    assert.ok(state.requests[0].output[210]);
    state.success('job-1', successfulText);
    state.emit('execution_error', {prompt_id: 'job-1', node_id: '234'});
    state.input.output[2000].inputs.value = false;
    await state.queue();
    const retry = state.requests[1];
    assert.equal(retry.output[210], undefined);
    assert.equal(retry.output[2000].inputs.value, false);
    assert.equal(retry.output[1605].inputs.value, successfulText);
    assert.deepEqual(Array.from(retry.output[2002].inputs.on_true), ['1605', 0]);
    assert.deepEqual(Array.from(retry.output[2002].inputs.on_false), ['1605', 0]);
    assert.equal(retry.output[2003].inputs.text, 'Prompt reused; OpenRouter disabled');
    const serializedCache = retry.workflow.nodes.find((node) => node.id === 1605);
    assert.equal(serializedCache.widgets_values[0], successfulText);
    assert.equal(serializedCache.widgets_values_named.value, successfulText);
    state.success('job-2', 'must not overwrite OFF cache');
    assert.equal(state.cacheNode.widgets[0].value, successfulText);
    assert.equal(state.graph.changed, 1);
});
