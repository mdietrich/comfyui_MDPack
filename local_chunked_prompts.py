import json
import math
import re
import time

try:
    from .openrouter_chunked_prompts import OpenRouterChunkedPrompts as _Shared
except ImportError:  # running the module directly (tests)
    from openrouter_chunked_prompts import OpenRouterChunkedPrompts as _Shared


class LocalChunkedPrompts:
    """
    Local counterpart of OpenRouterChunkedPrompts: generates an exact number
    of image prompts with a text encoder loaded through CLIPLoader (Qwen3-VL,
    Gemma 3) instead of an OpenRouter API call. Same input contract, same
    chunking, same diversity directives, same ||| output format, so the two
    nodes are interchangeable behind a Switch.

    Differences to the API node:
      - The chat template is built explicitly (system / user / assistant).
        The assistant turn is prefilled with the beginning of the JSON
        object, so a small local model cannot open with a preamble and the
        answer is parsed like the API answer. Truncated arrays are salvaged
        string by string.
      - max_tokens_per_call caps the generated tokens of one chunk; the
        chunk size is additionally limited so that the expected content fits
        into that budget.
      - The seed drives the diversity draw AND the sampler; every chunk uses
        seed + call index so the chunks differ from each other.
    """

    TEMPLATES = {
        "qwen (ChatML)": {
            "vision": "<|vision_start|><|image_pad|><|vision_end|>",
            "format": (
                "<|im_start|>system\n{system}<|im_end|>\n"
                "<|im_start|>user\n{vision}{user}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n{prefill}"
            ),
        },
        "gemma": {
            "vision": "\n<image_soft_token>\n",
            "format": (
                "<start_of_turn>user\n{system}\n\n{vision}{user}<end_of_turn>\n"
                "<start_of_turn>model\n{prefill}"
            ),
        },
    }

    PREFILL = '{"prompts": ["'
    TOKENS_PER_WORD = 1.6  # conservative for English prose incl. JSON escaping

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "Text encoder from CLIPLoader with a language "
                               "head (e.g. Qwen3-VL 4B/8B, type stable_diffusion).",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                }),
                "instruction": ("STRING", {"forceInput": True}),
                "total_count": ("INT", {
                    "default": 20, "min": 0, "max": 200,
                    "tooltip": "0 disables this node: it returns an empty "
                               "result without running the model.",
                }),
                "words_per_call": ("INT", {
                    "default": 600, "min": 100, "max": 5000,
                    "tooltip": "Content budget (words) per model call. Local "
                               "models lose quality on long outputs; keep small.",
                }),
                "expected_words_per_prompt": ("INT", {
                    "default": 0, "min": 0, "max": 2000,
                    "tooltip": "0 = auto-detect from the instruction text "
                               "(e.g. 'between 150 and 200 words')."
                }),
                "max_tokens_per_call": ("INT", {
                    "default": 4096, "min": 64, "max": 32768,
                    "tooltip": "Hard cap of generated tokens per call. The chunk "
                               "size is reduced so the expected content fits.",
                }),
                "template": (list(cls.TEMPLATES.keys()),),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.01, "max": 2.0,
                    "step": 0.05, "round": 0.01,
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "seed": ("INT", {
                    "forceInput": True, "default": 0,
                    "tooltip": "Drives the diversity draw and the sampler seed "
                               "(seed + call index per chunk).",
                }),
                "strict": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Raise an error if fewer than total_count "
                               "prompts were generated.",
                }),
                "diversity_axes": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "One axis per line: 'Label: option | option | "
                               "option'. Empty = built-in axes, '-' = no directives.",
                }),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("Prompts", "Stats",)
    FUNCTION = "generate"
    CATEGORY = "LLM"

    # ------------------------------------------------------------------ parsing

    @classmethod
    def clean_output(cls, text):
        text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
        text = re.sub(r"```(?:json)?", "", text)
        return text.strip()

    @classmethod
    def salvage_string_array(cls, text):
        """Read JSON string literals from the first '[' until the array
        breaks; returns whatever complete strings were found."""
        start = text.find("[")
        if start < 0:
            return []
        decoder = json.JSONDecoder()
        pos = start + 1
        found = []
        while True:
            while pos < len(text) and text[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(text) or text[pos] != '"':
                break
            try:
                value, end = decoder.raw_decode(text, pos)
            except json.JSONDecodeError:
                break
            if isinstance(value, str):
                found.append(value)
            pos = end
        return found

    @classmethod
    def extract_prompts(cls, raw, prefill=""):
        """Turn a model answer (with the prefill re-attached) into a list of
        prompt strings. Order of attempts: full JSON object, salvaged array,
        blank-line/numbered paragraphs."""
        text = cls.clean_output(prefill + raw)
        got = []
        brace = text.find("{")
        if brace >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text, brace)
                if isinstance(obj, dict):
                    arr = obj.get("prompts", [])
                elif isinstance(obj, list):
                    arr = obj
                else:
                    arr = []
                got = [str(p) for p in arr]
            except json.JSONDecodeError:
                got = cls.salvage_string_array(text[brace:])
        if not got:
            body = text[len(prefill):] if text.startswith(prefill) else text
            paragraphs = re.split(r"\n\s*\n|\n(?=\s*(?:\d+[.)]|-)\s)", body)
            got = [re.sub(r"^\s*(?:\d+[.)]|-)\s*", "", p) for p in paragraphs]
        cleaned = []
        for p in got:
            p = p.strip().strip('"').strip()
            if len(p.split()) >= 8:
                cleaned.append(p)
        return cleaned

    # --------------------------------------------------------------- generation

    @classmethod
    def build_chat(cls, template, system_prompt, user_text, has_image):
        spec = cls.TEMPLATES[template]
        return spec["format"].format(
            system=system_prompt.strip(),
            vision=spec["vision"] if has_image else "",
            user=user_text,
            prefill=cls.PREFILL,
        )

    def generate(self, clip, system_prompt, instruction, total_count,
                 words_per_call, expected_words_per_prompt, max_tokens_per_call,
                 template, temperature, image=None, seed=0, strict=False,
                 diversity_axes="", top_p=0.95, repetition_penalty=1.05):

        if int(total_count) == 0:
            return ("", "disabled: total_count=0")

        if int(expected_words_per_prompt) > 0:
            expected = float(expected_words_per_prompt)
            expected_src = "manual"
        else:
            detected = _Shared.detect_expected_words(instruction)
            expected = detected if detected else 175.0
            expected_src = "auto-detected" if detected else "fallback"

        # Two limits on the chunk: the word budget and the token cap.
        chunk_by_words = max(1, int(max(100, int(words_per_call)) // expected))
        tokens_per_prompt = expected * self.TOKENS_PER_WORD + 16
        chunk_by_tokens = max(1, int((int(max_tokens_per_call) - 64) // tokens_per_prompt))
        chunk = max(1, min(chunk_by_words, chunk_by_tokens))

        axes = _Shared.parse_diversity_axes(diversity_axes)
        total = max(1, int(total_count))
        max_calls = math.ceil(total / chunk) * 2 + 2
        prompts = []
        stats = [
            f"chunk_size={chunk} (words: {chunk_by_words}, tokens: {chunk_by_tokens}; "
            f"{expected_src}: ~{expected:.0f} words/prompt, budget "
            f"{int(words_per_call)} words / {int(max_tokens_per_call)} tokens per call)",
            f"diversity: {len(axes)} axes, seed={int(seed)}, template={template}, "
            f"image={'yes' if image is not None else 'no'}",
        ]
        print(f"LocalChunkedPrompts: {stats[0]}")
        calls = 0

        while len(prompts) < total and calls < max_calls:
            n = min(chunk, total - len(prompts))
            text = instruction.replace("<image_count>", str(n))
            if total > n:
                done = len(prompts)
                text += (
                    f"\n\nSERIES CONTEXT: You are now generating prompts "
                    f"{done + 1} through {done + n} of {total} total prompts. "
                    f"Any distribution, variety, or percentage requirements in "
                    f"the instructions above apply across the whole series of "
                    f"{total} prompts, so cover a representative share of that "
                    f"range in this batch."
                )
            if prompts:
                seen = "\n- ".join(_Shared.prompt_signature(p) for p in prompts)
                text += (
                    "\n\nYou have already generated prompts starting as listed "
                    "below in previous batches. The new prompts must be clearly "
                    "different from ALL of them: use different poses, "
                    "activities, and compositions. Do not repeat or closely "
                    "paraphrase any of them.\nAlready generated:\n- " + seen
                )
            text += _Shared.diversity_directives(axes, seed, len(prompts) + 1, n, total)
            text += _Shared.output_contract(n)

            chat = self.build_chat(template, system_prompt, text, image is not None)
            call_seed = (int(seed) + calls) & 0xFFFFFFFFFFFFFFFF
            max_new = min(int(max_tokens_per_call), int(n * tokens_per_prompt + 64))

            calls += 1
            start = time.time()
            got = []
            note = ""
            try:
                tokens = clip.tokenize(chat, image=image, skip_template=True, min_length=1)
                generated_ids = clip.generate(
                    tokens, do_sample=True, max_length=max_new,
                    temperature=float(temperature), top_k=64, top_p=float(top_p),
                    min_p=0.05, repetition_penalty=float(repetition_penalty),
                    seed=call_seed,
                )
                raw = clip.decode(generated_ids)
                got = [_Shared.scrub_leaks(p) for p in self.extract_prompts(raw, self.PREFILL)]
                got = [p for p in got if p]
                note = f", tokens<={max_new}, chars={len(raw)}"
                if not got:
                    note += f", BAD CONTENT: head={raw[:120]!r}"
            except Exception as e:
                note = f", ERROR: {e}"

            prompts.extend(got)
            stats.append(
                f"call {calls}: requested {n}, received {len(got)} "
                f"(total {len(prompts)}/{total}), {time.time() - start:.1f}s{note}"
            )
            print(f"LocalChunkedPrompts: {stats[-1]}")

        if not prompts:
            raise RuntimeError(
                "LocalChunkedPrompts: no prompts generated after "
                f"{calls} calls:\n" + "\n".join(stats)
            )
        if len(prompts) < total:
            stats.append(f"WARNING: only {len(prompts)}/{total} prompts after {calls} calls")
            print(f"LocalChunkedPrompts: {stats[-1]}")
            if strict:
                raise RuntimeError(
                    f"LocalChunkedPrompts: strict mode - only "
                    f"{len(prompts)}/{total} prompts after {calls} calls:\n"
                    + "\n".join(stats)
                )

        prompts = prompts[:total]
        return ("|||".join(prompts), "\n".join(stats))


NODE_CLASS_MAPPINGS = {
    "LocalChunkedPrompts": LocalChunkedPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalChunkedPrompts": "Local Chunked Prompts (CLIP LLM)",
}
