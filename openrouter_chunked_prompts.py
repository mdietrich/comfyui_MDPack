import base64
import io
import json
import math
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
    guarantees the total count. Each call uses JSON mode and receives the
    previously generated prompts as a do-not-repeat list to keep variety.

    The chunk size adapts to the requested prompt length: the expected words
    per prompt are parsed from the instruction (e.g. "between 150 and 200
    words", "roughly 150 words"); words_per_call is the content budget one
    call may carry, so chunk = words_per_call / expected words. Set
    expected_words_per_prompt > 0 to override the auto-detection.

    The instruction input must contain the placeholder <image_count>; it is
    replaced per call with the chunk size.

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
                "total_count": ("INT", {"default": 20, "min": 1, "max": 200}),
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

    @classmethod
    def detect_expected_words(cls, instruction):
        for pattern, extract in cls.WORD_PATTERNS:
            m = pattern.search(instruction)
            if m:
                return extract(m)
        return None

    def generate(self, api_key, system_prompt, instruction, total_count,
                 words_per_call, expected_words_per_prompt, model, temperature,
                 image=None):
        if not api_key:
            raise ValueError("OpenRouterChunkedPrompts: API key not provided.")
        if "<image_count>" not in instruction:
            print("OpenRouterChunkedPrompts: warning - instruction contains no "
                  "<image_count> placeholder; chunk sizes cannot be injected.")

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

        total = max(1, int(total_count))
        chunk = max(1, int(chunk_size))
        max_calls = math.ceil(total / chunk) * 2 + 2
        prompts = []
        stats = [
            f"chunk_size={chunk} ({expected_src}: ~{expected:.0f} words/prompt, "
            f"budget {int(words_per_call)} words/call)"
        ]
        print(f"OpenRouterChunkedPrompts: {stats[0]}")
        calls = 0

        while len(prompts) < total and calls < max_calls:
            n = min(chunk, total - len(prompts))
            text = instruction.replace("<image_count>", str(n))
            if prompts:
                seen = "\n- ".join(p[:250] for p in prompts)
                text += (
                    "\n\nYou have already generated the prompts listed below in previous "
                    "batches. The new prompts must be clearly different from ALL of them: "
                    "use different poses, activities, and compositions. Do not repeat or "
                    "closely paraphrase any of them.\nAlready generated:\n- " + seen
                )

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
                        got = [str(p).strip() for p in arr if str(p).strip()]
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

        prompts = prompts[:total]
        return ("|||".join(prompts), "\n".join(stats))


NODE_CLASS_MAPPINGS = {
    "OpenRouterChunkedPrompts": OpenRouterChunkedPrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenRouterChunkedPrompts": "OpenRouter Chunked Prompts",
}
