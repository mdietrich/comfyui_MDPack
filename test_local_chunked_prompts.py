"""Tests for LocalChunkedPrompts with a fake CLIP (no model needed).

Run remotely:
    ssh gamingmonster '~/ComfyUI_winows_portable/.venv/bin/python \
        ~/ComfyUI_winows_portable/ComfyUI/custom_nodes/comfyui_MDPack/test_local_chunked_prompts.py'
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_chunked_prompts as lcp  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


LONG = "a woman standing in a sunlit kitchen holding a white ceramic mug near the window"


class FakeClip:
    """Answers every call with `per_call` prompts, JSON continuing the prefill."""

    def __init__(self, per_call, answer=None):
        self.per_call = per_call
        self.answer = answer
        self.calls = []

    def tokenize(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return {"fake": text}

    def generate(self, tokens, **kwargs):
        self.calls[-1]["gen"] = kwargs
        return tokens

    def decode(self, tokens, skip_special_tokens=True):
        if self.answer is not None:
            return self.answer
        prompt_no = len(self.calls)
        items = [f"{LONG} variant {prompt_no}-{i}" for i in range(self.per_call)]
        # continuation of the prefill '{"prompts": ["'
        return json.dumps({"prompts": items})[len(lcp.LocalChunkedPrompts.PREFILL):]


NODE = lcp.LocalChunkedPrompts()
KW = dict(system_prompt="sys", words_per_call=600, expected_words_per_prompt=0,
          max_tokens_per_call=4096, template="qwen (ChatML)", temperature=0.8)

# --- disabled path
r = NODE.generate(clip=FakeClip(1), instruction="x", total_count=0, **KW)
check("total_count=0 short-circuits", r == ("", "disabled: total_count=0"), r)

# --- parsing
P = lcp.LocalChunkedPrompts.PREFILL
check("parse full json",
      lcp.LocalChunkedPrompts.extract_prompts(json.dumps({"prompts": [LONG, LONG + " two"]})[len(P):], P)
      == [LONG, LONG + " two"])
truncated = json.dumps({"prompts": [LONG, LONG + " two", LONG + " three"]})[len(P):]
truncated = truncated[:truncated.rfind('"') - 5]  # cut inside the third string
check("salvage truncated array",
      lcp.LocalChunkedPrompts.extract_prompts(truncated, P) == [LONG, LONG + " two"],
      lcp.LocalChunkedPrompts.extract_prompts(truncated, P))
check("strip think block",
      lcp.LocalChunkedPrompts.extract_prompts(
          "<think>hmm</think>" + json.dumps({"prompts": [LONG]})[len(P):], P) == [LONG])
plain = f"1. {LONG} one\n\n2. {LONG} two\n"
check("fallback numbered paragraphs",
      lcp.LocalChunkedPrompts.extract_prompts(plain, "") == [LONG + " one", LONG + " two"],
      lcp.LocalChunkedPrompts.extract_prompts(plain, ""))

# --- chunking with 300-400 word prompts: 600 words/call -> 1 prompt per call
clip = FakeClip(1)
instr = "Write <image_count> prompts. Every prompt must be between 300 and 400 words long."
out, stats = NODE.generate(clip=clip, instruction=instr, total_count=3, seed=7,
                           diversity_axes="-", **KW)
check("3 prompts from 3 calls", out.count("|||") == 2 and len(clip.calls) == 3,
      f"calls={len(clip.calls)} out={out[:80]}")
check("<image_count> replaced per chunk", "generating prompts 2 through 2 of 3" in clip.calls[1]["text"])
check("chat template has system + prefill",
      clip.calls[0]["text"].startswith("<|im_start|>system\nsys<|im_end|>")
      and clip.calls[0]["text"].endswith(P))
check("no vision block without image", "<|vision_start|>" not in clip.calls[0]["text"])
check("seed advances per call", [c["gen"]["seed"] for c in clip.calls] == [7, 8, 9])
check("already-generated list on 2nd call", "Already generated:" in clip.calls[1]["text"])
check("skip_template passed", clip.calls[0].get("skip_template") is True)

# --- image adds vision block and is forwarded
clip = FakeClip(2)
out, _ = NODE.generate(clip=clip, instruction="Write <image_count> prompts of roughly 100 words.",
                       total_count=2, seed=1, image="IMG", **KW)
check("single call for 2 short prompts", len(clip.calls) == 1 and out.count("|||") == 1)
check("vision block present with image", "<|vision_start|><|image_pad|><|vision_end|>" in clip.calls[0]["text"])
check("image forwarded to tokenize", clip.calls[0].get("image") == "IMG")

# --- token cap limits chunk size
clip = FakeClip(1)
out, stats = NODE.generate(clip=clip, instruction="Write <image_count> prompts of roughly 100 words.",
                           total_count=2, seed=1, **{**KW, "max_tokens_per_call": 300})
check("token cap forces 1 prompt/call", len(clip.calls) == 2, stats)

# --- strict
clip = FakeClip(1, answer="nothing useful")
try:
    NODE.generate(clip=clip, instruction="x", total_count=1, strict=True, **KW)
    check("strict raises on empty", False)
except RuntimeError:
    check("strict raises on empty", True)

# --- gemma template
chat = lcp.LocalChunkedPrompts.build_chat("gemma", "S", "U", True)
check("gemma template", chat.startswith("<start_of_turn>user\nS\n\n\n<image_soft_token>\nU") and chat.endswith(P))

print()
print("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}")
sys.exit(1 if FAILURES else 0)
