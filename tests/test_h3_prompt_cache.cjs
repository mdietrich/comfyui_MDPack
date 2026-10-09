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

function environment({enabled = false, cached = '  saved\nexact text\n', scoped = true, final = false, early = false, deferred = false} = {}) {
    const cacheNode = {widgets: [{name: 'value', value: cached}], widgets_values_named: {value: cached}, setDirtyCanvas() {}};
    const graph = {extra: scoped ? {h3_prompt_cache: {...marker}} : {}, changed: 0,
        getNodeById(nodeId) { return Number(nodeId) === 1605 ? cacheNode : null; }, change() { this.changed++; }};
    const listeners = new Map();
    const requests = [];
    const queueResolvers = [];
    const input = {workflow: {extra: structuredClone(graph.extra)}, output: {
        210: {class_type: 'OpenRouterNode', inputs: {user_message_input: 'unchanged request', system_prompt: 'unchanged system'}},
        211: {class_type: 'ShowText|pysssss', inputs: {text: ['2002', 0]}},
        233: {class_type: 'MiniMaxH3PromptBuilder', inputs: {prompt_text: ['211', 0]}},
        2000: {class_type: 'PrimitiveBoolean', inputs: {value: enabled}},
        1605: {class_type: 'PrimitiveStringMultiline', inputs: {value: cached}},
        2002: {class_type: 'LazySwitch (obvpm)', inputs: {boolean: ['2000', 0], on_true: ['210', 0], on_false: ['1605', 0]}},
        2003: {class_type: 'ShowText|pysssss', inputs: {text: ['210', 2]}},
    }};
    if (final) { delete input.output[211]; delete input.output[2003]; }
    const application = {graph, registerExtension() {}, async graphToPrompt() { return structuredClone(input); }};
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
    return {application, serverApi, graph, cacheNode, requests, input, controller, browserCrypto,
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
