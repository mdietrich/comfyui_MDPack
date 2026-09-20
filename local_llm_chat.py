import re
import time


class LocalLLMChat:
    """
    Single-shot chat call against a local text encoder loaded through
    CLIPLoader (Qwen3-VL, Gemma 3, ...). Takes a system prompt and a user
    prompt, builds the chat template, runs one generation, and returns the
    raw model answer.

    Model selection happens upstream: connect the CLIP output of a
    CLIPLoader node (or any node providing a CLIP with a language head) to
    the clip input here.

    Qwen3(-VL) is a reasoning model: without a forced empty <think></think>
    block it can drift into a degenerate loop on some quants/model sizes
    (observed as endless "," output on 32B convrot checkpoints). The qwen
    template therefore always closes the think block before the answer,
    matching the working pattern from LocalChunkedPrompts/OpenRouterChunkedPrompts.
    """

    TEMPLATES = {
        "qwen (ChatML)": {
            "vision": "<|vision_start|><|image_pad|><|vision_end|>",
            "format": (
                "<|im_start|>system\n{system}<|im_end|>\n"
                "<|im_start|>user\n{vision}{user}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
        },
        "gemma": {
            "vision": "\n<image_soft_token>\n",
            "format": (
                "<start_of_turn>system\n{system}<end_of_turn>\n"
                "<start_of_turn>user\n{vision}{user}<end_of_turn>\n"
                "<start_of_turn>model\n"
            ),
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "Text encoder from CLIPLoader with a language "
                               "head (e.g. Qwen3-VL 4B/8B, type stable_diffusion). "
                               "Model choice happens on that node, not here.",
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                }),
                "template": (list(cls.TEMPLATES.keys()),),
                "max_tokens": ("INT", {
                    "default": 1024, "min": 1, "max": 32768,
                    "tooltip": "Hard cap of generated tokens.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0,
                    "step": 0.05, "round": 0.01,
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "seed": ("INT", {
                    "forceInput": True, "default": 0,
                }),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("Response", "Stats",)
    FUNCTION = "generate"
    CATEGORY = "LLM"

    @staticmethod
    def clean_output(text):
        return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()

    def build_chat(self, template, system_prompt, user_prompt, has_image):
        spec = self.TEMPLATES[template]
        return spec["format"].format(
            system=system_prompt.strip(),
            vision=spec["vision"] if has_image else "",
            user=user_prompt,
        )

    def generate(self, clip, user_prompt, system_prompt, template, max_tokens,
                 temperature, image=None, seed=0, top_p=0.95, repetition_penalty=1.05):

        chat = self.build_chat(template, system_prompt, user_prompt, image is not None)
        do_sample = float(temperature) > 0.0

        start = time.time()
        tokens = clip.tokenize(chat, image=image, skip_template=True, min_length=1)
        generated_ids = clip.generate(
            tokens, do_sample=do_sample, max_length=int(max_tokens),
            temperature=max(float(temperature), 0.01), top_k=64, top_p=float(top_p),
            min_p=0.05, repetition_penalty=float(repetition_penalty),
            seed=int(seed),
        )
        raw = self.clean_output(clip.decode(generated_ids))
        elapsed = time.time() - start

        stats = (
            f"template={template}, image={'yes' if image is not None else 'no'}, "
            f"max_tokens={int(max_tokens)}, temperature={float(temperature)}, "
            f"seed={int(seed)}, chars={len(raw)}, {elapsed:.1f}s"
        )
        print(f"LocalLLMChat: {stats}")
        return (raw, stats)


NODE_CLASS_MAPPINGS = {
    "LocalLLMChat": LocalLLMChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalLLMChat": "Local LLM Chat (CLIP LLM)",
}
