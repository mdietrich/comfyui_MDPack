import base64
import io
import json
import math
import random
import re
import time

import numpy as np
import requests
from PIL import Image


class OpenRouterChunkedPrompts:
    """
    Generates a large, exact number of image prompts by splitting the request
    into several smaller OpenRouter chat-completion calls ("chunks").

    Rationale: reasoning models (e.g. grok-4.3) have a soft output budget and
    silently deliver fewer/shorter items when asked for many long prompts at
    once. Small chunks keep every prompt at full length while the node
    guarantees the total count. Each call uses JSON mode and receives short
    signatures of the previously generated prompts as a do-not-repeat list to
    keep variety.

    The node owns the output contract: it appends the CRITICAL OUTPUT
    REQUIREMENT block (exact JSON shape, exact per-call count) as the very
    last part of every call, so the instruction text does not need to state
    it. If the instruction contains the <image_count> placeholder it is still
    replaced per call with the chunk size for backwards compatibility.

    When the request is split into several calls, each call also receives a
    series context ("prompts i through j of N total") so that distribution
    or percentage requirements in the instruction are understood as applying
    to the whole series, not just the current chunk.

    The chunk size adapts to the requested prompt length: the expected words
    per prompt are parsed from the instruction (e.g. "between 150 and 200
    words", "roughly 150 words"); words_per_call is the content budget one
    call may carry, so chunk = words_per_call / expected words. Set
    expected_words_per_prompt > 0 to override the auto-detection.

    Identical requests make a model fall back to its favourite scenes, so
    repeated runs deliver near-identical batches. To break that, the node
    draws one option per diversity axis per prompt, deterministically from
    the seed, and appends the drawn combination as a per-prompt directive
    line. A different seed therefore yields a genuinely different batch.

    Optional inputs:
      - seed: drives the diversity draw and forces re-execution. The prompt
        text itself is never sent as a number to the model; only the drawn
        axis values and a variation marker are.
      - diversity_axes: one axis per line as "Label: option | option | ...".
        Empty falls back to DEFAULT_DIVERSITY_AXES; a single "-" disables
        the directives entirely.
      - strict: raise an error when fewer than total_count prompts could be
        generated instead of returning a short result with a warning.

    Outputs:
      1) "Prompts": all prompts joined with the ||| sentinel delimiter
      2) "Stats": per-call diagnostics (counts, tokens, timing)
    """

    models_cache = None
    last_fetch_time = 0
    cache_duration = 3600

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant."
                }),
                "instruction": ("STRING", {"forceInput": True}),
                "total_count": ("INT", {
                    "default": 20, "min": 0, "max": 200,
                    "tooltip": "0 disables this node: it returns an empty "
                               "result without calling the API.",
                }),
                "words_per_call": ("INT", {
                    "default": 900, "min": 100, "max": 5000,
                    "tooltip": "Content budget (words) per API call; keeps each "
                               "call safely below the model's soft output budget."
                }),
                "expected_words_per_prompt": ("INT", {
                    "default": 0, "min": 0, "max": 2000,
                    "tooltip": "0 = auto-detect from the instruction text "
                               "(e.g. 'between 150 and 200 words')."
                }),
                "model": (cls.fetch_openrouter_models(),),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0,
                    "step": 0.1, "display": "slider", "round": 1,
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "seed": ("INT", {
                    "forceInput": True, "default": 0,
                    "tooltip": "Cache-busting only: forces re-execution on "
                               "change, never sent to the LLM.",
                }),
                "strict": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Raise an error if fewer than total_count "
                               "prompts were generated.",
                }),
                "diversity_axes": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "One axis per line: 'Label: option | option | "
                               "option'. One option per axis is drawn per "
                               "prompt from the seed and handed to the model "
                               "as a directive. Empty = built-in axes, "
                               "'-' = no directives.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("Prompts", "Stats",)
    FUNCTION = "generate"
    CATEGORY = "LLM"

    @classmethod
    def fetch_openrouter_models(cls):
        current_time = time.time()
        if (cls.models_cache is None) or (current_time - cls.last_fetch_time > cls.cache_duration):
            try:
                response = requests.get("https://openrouter.ai/api/v1/models", timeout=20)
                response.raise_for_status()
                models = response.json()["data"]
                cls.models_cache = sorted([model["id"] for model in models])
                cls.last_fetch_time = current_time
            except requests.exceptions.RequestException as e:
                print(f"OpenRouterChunkedPrompts: error fetching models: {e}")
                if cls.models_cache is None:
                    cls.models_cache = ["x-ai/grok-4.3"]
        return cls.models_cache if cls.models_cache else ["x-ai/grok-4.3"]

    def image_to_base64(self, image_tensor):
        image_np = image_tensor.cpu().numpy() if hasattr(image_tensor, "cpu") else np.asarray(image_tensor)
        if image_np.ndim == 4:
            image_np = image_np[0]
        image_np = (np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    WORD_PATTERNS = [
        (re.compile(r"between\s+(\d+)\s+and\s+(\d+)\s+words", re.I),
         lambda m: (int(m.group(1)) + int(m.group(2))) / 2),
        (re.compile(r"(\d+)\s*(?:-|to)\s*(\d+)\s+words", re.I),
         lambda m: (int(m.group(1)) + int(m.group(2))) / 2),
        (re.compile(r"(?:roughly|approximately|about|around|~)\s*(\d+)\s+words", re.I),
         lambda m: float(m.group(1))),
        (re.compile(r"at\s+least\s+(\d+)\s+words", re.I),
         lambda m: int(m.group(1)) * 1.2),
    ]

    SIGNATURE_WORDS = 18

    DEFAULT_DIVERSITY_AXES = (
        "Setting: indoor domestic | indoor public | workplace or studio | "
        "outdoor urban | outdoor nature | transit or vehicle | "
        "water or poolside | nightlife or event\n"
        "Time of day: early morning | midday | afternoon | golden hour | "
        "blue hour | night\n"
        "Light: hard direct sunlight | overcast diffuse | window light | "
        "practical lamps | neon or colored light | camera flash\n"
        "Framing: wide establishing shot | full body | medium shot | "
        "close-up | over-the-shoulder | low angle | high angle\n"
        "Energy: still and calm | casual everyday action | focused activity | "
        "dynamic movement"
    )

    @classmethod
    def parse_diversity_axes(cls, raw):
        """Parse "Label: a | b | c" lines into [(label, [options])]."""
        text = (raw or "").strip()
        if text == "-":
            return []
        if not text:
            text = cls.DEFAULT_DIVERSITY_AXES
        axes = []
        for line in text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            label, _, options_part = line.partition(":")
            options = [o.strip() for o in options_part.split("|") if o.strip()]
            if label.strip() and options:
                axes.append((label.strip(), options))
        return axes

    @classmethod
    def diversity_directives(cls, axes, seed, first_index, count, total):
        """Draw one option per axis per prompt, deterministic in (seed, index).

        Returns an empty string when no axes are configured: the seed alone
        already busts the cache, and any marker text (e.g. a seed number)
        risks being copied verbatim into the prompts and rendered as text
        in the image.
        """
        if not axes:
            return ""
        lines = []
        for offset in range(count):
            index = first_index + offset
            rng = random.Random(f"{int(seed)}:{index}")
            drawn = "; ".join(
                f"{label} = {rng.choice(options)}" for label, options in axes
            )
            lines.append(f"- prompt {offset + 1}: {drawn}")
        return (
            "\n\nPER-PROMPT DIRECTIVES (internal planning notes, not prompt "
            "content): every prompt in this batch must follow its own line "
            "below. The directives fix those axes; everything else stays your "
            "creative choice. They override your default preferences, so do "
            "not fall back to recurring favourite scenes. Never quote these "
            "notes, their labels, or any numbers in the prompt text itself; "
            "express the drawn values only as natural visual description:\n"
            + "\n".join(lines)
        )

    LEAK_PATTERN = re.compile(
        r"[\s,;:(-]*\b(?:variation\s+(?:marker|directives?)|per-prompt\s+directives?)\b"
        r"[^.\n]*(?:\.|(?=\n)|$)",
        re.IGNORECASE,
    )

    @classmethod
    def scrub_leaks(cls, prompt):
        """Remove directive wording a model copied into a prompt."""
        cleaned = cls.LEAK_PATTERN.sub(" ", prompt)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        return cleaned

    @classmethod
    def detect_expected_words(cls, instruction):
        for pattern, extract in cls.WORD_PATTERNS:
            m = pattern.search(instruction)
            if m:
                return extract(m)
        return None

    @classmethod
    def prompt_signature(cls, prompt):
        words = prompt.split()
        signature = " ".join(words[:cls.SIGNATURE_WORDS])
        if len(words) > cls.SIGNATURE_WORDS:
            signature += " ..."
        return signature

    @staticmethod
    def output_contract(n):
        return (
            "\n\nCRITICAL OUTPUT REQUIREMENT: Respond with ONLY a valid JSON "
            'object of the form {"prompts": ["...", "..."]} where the array '
            f"contains EXACTLY {n} prompt strings. An array with fewer than "
            f"{n} entries is a failed response. Do not stop early; only "
            f"finish after prompt number {n} is complete."
        )

    def generate(self, api_key, system_prompt, instruction, total_count,
                 words_per_call, expected_words_per_prompt, model, temperature,
                 image=None, seed=0, strict=False, diversity_axes=""):

        # A disabled node must not touch the network or require credentials;
        # the workflow uses total_count=0 to switch off a whole prompt branch.
        if int(total_count) == 0:
            return ("", "disabled: total_count=0")

        if not api_key:
            raise ValueError("OpenRouterChunkedPrompts: API key not provided.")

        if int(expected_words_per_prompt) > 0:
            expected = float(expected_words_per_prompt)
            expected_src = "manual"
        else:
            detected = self.detect_expected_words(instruction)
            expected = detected if detected else 175.0
            expected_src = "auto-detected" if detected else "fallback"
        chunk_size = max(1, int(max(100, int(words_per_call)) // expected))

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/openrouter-chunked",
            "X-Title": "ComfyUI OpenRouter Chunked Prompts",
        }

        image_b64 = self.image_to_base64(image) if image is not None else None
        axes = self.parse_diversity_axes(diversity_axes)

        total = max(1, int(total_count))
        chunk = max(1, int(chunk_size))
        max_calls = math.ceil(total / chunk) * 2 + 2
        prompts = []
        stats = [
            f"chunk_size={chunk} ({expected_src}: ~{expected:.0f} words/prompt, "
            f"budget {int(words_per_call)} words/call)",
            f"diversity: {len(axes)} axes, seed={int(seed)}",
        ]
        print(f"OpenRouterChunkedPrompts: {stats[0]}")
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
                seen = "\n- ".join(self.prompt_signature(p) for p in prompts)
                text += (
                    "\n\nYou have already generated prompts starting as listed "
                    "below in previous batches. The new prompts must be clearly "
                    "different from ALL of them: use different poses, "
                    "activities, and compositions. Do not repeat or closely "
                    "paraphrase any of them.\nAlready generated:\n- " + seen
                )
            text += self.diversity_directives(
                axes, seed, len(prompts) + 1, n, total)
            text += self.output_contract(n)

            content_blocks = [{"type": "text", "text": text}]
            if image_b64 is not None:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                })

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": content_blocks if image_b64 is not None else text},
                ],
                "temperature": float(temperature),
                "response_format": {"type": "json_object"},
            }
            if int(seed):
                # Providers that honour it sample differently per seed; those
                # that ignore it are already covered by the directives above.
                payload["seed"] = int(seed) & 0x7FFFFFFF

            calls += 1
            start = time.time()
            got = []
            usage_note = ""
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=600)
                response.raise_for_status()
                result = response.json()
                if not result.get("choices"):
                    usage_note = f", API-ERROR: {str(result.get('error') or result)[:200]}"
                else:
                    choice = result["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "") or ""
                    usage = result.get("usage", {})
                    usage_note = f", completion_tokens={usage.get('completion_tokens')}"
                    try:
                        arr = json.loads(content).get("prompts", [])
                        got = [self.scrub_leaks(str(p)) for p in arr if str(p).strip()]
                        got = [p for p in got if p]
                    except Exception:
                        usage_note += (
                            f", BAD CONTENT: finish={choice.get('finish_reason')}"
                            f"/{choice.get('native_finish_reason')}"
                            f", refusal={str(message.get('refusal'))[:80]}"
                            f", len={len(content)}, head={content[:120]!r}"
                        )
            except Exception as e:
                usage_note = f", ERROR: {e}"

            prompts.extend(got)
            stats.append(
                f"call {calls}: requested {n}, received {len(got)} "
                f"(total {len(prompts)}/{total}), {time.time() - start:.1f}s{usage_note}"
            )
            print(f"OpenRouterChunkedPrompts: {stats[-1]}")

        if not prompts:
            raise RuntimeError(
                "OpenRouterChunkedPrompts: no prompts generated after "
                f"{calls} calls:\n" + "\n".join(stats)
            )
        if len(prompts) < total:
            stats.append(f"WARNING: only {len(prompts)}/{total} prompts after {calls} calls")
            print(f"OpenRouterChunkedPrompts: {stats[-1]}")
            if strict:
                raise RuntimeError(
                    f"OpenRouterChunkedPrompts: strict mode - only "
                    f"{len(prompts)}/{total} prompts after {calls} calls:\n"
                    + "\n".join(stats)
                )

        prompts = prompts[:total]
        return ("|||".join(prompts), "\n".join(stats))


NODE_CLASS_MAPPINGS = {
    "OpenRouterChunkedPrompts": OpenRouterChunkedPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterChunkedPrompts": "OpenRouter Chunked Prompts",
}
