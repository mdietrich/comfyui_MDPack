"""Tests for the OpenRouterChunkedPrompts disable path.

Run remotely:
    ssh gamingmonster '~/ComfyUI_winows_portable/.venv/bin/python \
        ~/ComfyUI_winows_portable/ComfyUI/custom_nodes/comfyui_MDPack/test_openrouter_chunked_prompts.py'
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openrouter_chunked_prompts as ocp  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


def exploding_post(*args, **kwargs):
    raise AssertionError("network call issued although the node was disabled")


NODE = ocp.OpenRouterChunkedPrompts()

# total_count = 0 must short-circuit before the API key check and before any request.
ORIGINAL_POST = ocp.requests.post
ocp.requests.post = exploding_post
try:
    result = NODE.generate(
        api_key="", system_prompt="sys", instruction="write prompts",
        total_count=0, words_per_call=4000, expected_words_per_prompt=0,
        model="x-ai/grok-4.3", temperature=1.0, strict=True,
    )
    check("total_count=0 returns empty prompts and a disabled marker",
          result == ("", "disabled: total_count=0"), str(result))
except Exception as error:
    check(f"total_count=0 raised instead of returning ({error})", False)
finally:
    ocp.requests.post = ORIGINAL_POST

# The widget must actually allow 0, otherwise the workflow cannot send it.
TOTAL_COUNT_SPEC = ocp.OpenRouterChunkedPrompts.INPUT_TYPES()["required"]["total_count"][1]
check("total_count widget allows 0", TOTAL_COUNT_SPEC["min"] == 0, str(TOTAL_COUNT_SPEC))

# A positive total_count must still take the normal path (missing key -> error).
try:
    NODE.generate(
        api_key="", system_prompt="sys", instruction="write prompts",
        total_count=3, words_per_call=4000, expected_words_per_prompt=0,
        model="x-ai/grok-4.3", temperature=1.0,
    )
    check("total_count=3 without api_key still raises", False, "no error raised")
except ValueError as error:
    check("total_count=3 without api_key still raises", "API key" in str(error), str(error))

# --- diversity directives ------------------------------------------------

CLASS = ocp.OpenRouterChunkedPrompts

# The new widget must sit AFTER every existing one: widgets_values is
# positional, so inserting earlier would shift saved workflows.
OPTIONAL_KEYS = list(CLASS.INPUT_TYPES()["optional"].keys())
check("diversity_axes is the last optional input",
      OPTIONAL_KEYS[-1] == "diversity_axes", str(OPTIONAL_KEYS))

parsed = CLASS.parse_diversity_axes("Setting: kitchen | street\nLight: sun | neon")
check("axes parse into label/option pairs",
      parsed == [("Setting", ["kitchen", "street"]), ("Light", ["sun", "neon"])],
      str(parsed))
check("empty axes fall back to the built-ins",
      len(CLASS.parse_diversity_axes("")) == 5,
      str(CLASS.parse_diversity_axes("")))
check("'-' disables the axes", CLASS.parse_diversity_axes("-") == [])

AXES = CLASS.parse_diversity_axes("")
FIRST = CLASS.diversity_directives(AXES, 111, 1, 2, 2)
SECOND = CLASS.diversity_directives(AXES, 222, 1, 2, 2)
check("same seed is reproducible",
      FIRST == CLASS.diversity_directives(AXES, 111, 1, 2, 2))
check("different seed yields different directives", FIRST != SECOND)
check("one directive line per prompt",
      FIRST.count("- prompt ") == 2, FIRST)
check("prompts inside one batch differ",
      FIRST.split("- prompt 1:")[1].split("\n")[0]
      != FIRST.split("- prompt 2:")[1].split("\n")[0], FIRST)

# Over many seeds the drawn setting must actually spread across the axis.
SETTINGS = {
    CLASS.diversity_directives(AXES, seed, 1, 1, 1).split("Setting = ")[1].split(";")[0]
    for seed in range(200)
}
check("the setting axis is actually exercised", len(SETTINGS) >= 6, str(SETTINGS))

check("no axes injects nothing (no marker text that could leak into images)",
      CLASS.diversity_directives([], 7, 1, 2, 2) == "")
HEADER = CLASS.diversity_directives(AXES, 4242, 1, 1, 1).split("\n- prompt 1")[0]
check("directive header carries no seed number", "4242" not in HEADER and "seed" not in HEADER.lower())
check("scrub removes leaked marker sentence",
      CLASS.scrub_leaks("warm light on the deck. VARIATION MARKER 13665587 this batch must not repeat scenes. She smiles.")
      == "warm light on the deck. She smiles.")
check("scrub removes inline marker phrase",
      "13665587" not in CLASS.scrub_leaks("subtle film grain in this VARIATION MARKER 13665587 scene. Next sentence."))
check("scrub keeps clean prompts untouched", CLASS.scrub_leaks("A calm portrait.") == "A calm portrait.")

# The seed must reach the request payload as well.
CAPTURED = {}


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": '{"prompts": ["a scene"]}'}}],
                "usage": {"completion_tokens": 5}}


def capturing_post(url, headers=None, json=None, timeout=None):
    CAPTURED.update(json)
    return FakeResponse()


ocp.requests.post = capturing_post
try:
    PROMPTS, STATS = NODE.generate(
        api_key="key", system_prompt="sys", instruction="write prompts",
        total_count=1, words_per_call=4000, expected_words_per_prompt=0,
        model="x-ai/grok-4.3", temperature=1.0, seed=4242,
    )
    SENT = CAPTURED["messages"][1]["content"]
    check("directives reach the user message", "PER-PROMPT DIRECTIVES" in SENT, SENT[-300:])
    check("seed reaches the payload", CAPTURED.get("seed") == 4242, str(CAPTURED.get("seed")))
    check("stats report the diversity setup", "diversity: 5 axes" in STATS, STATS)
    CAPTURED.clear()
    NODE.generate(
        api_key="key", system_prompt="sys", instruction="write prompts",
        total_count=1, words_per_call=4000, expected_words_per_prompt=0,
        model="x-ai/grok-4.3", temperature=1.0, seed=0,
    )
    check("seed=0 stays out of the payload", "seed" not in CAPTURED, str(CAPTURED.keys()))
finally:
    ocp.requests.post = ORIGINAL_POST

print()
if FAILURES:
    print(f"{len(FAILURES)} TEST(S) FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("ALL TESTS PASSED")
