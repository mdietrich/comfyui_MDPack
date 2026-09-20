import os
import time

import folder_paths

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None


# Folders scanned for HF model directories (anything with a config.json).
MODEL_DIRS = ("prompt_generator", "LLM")


def discover_models():
    found = []
    for sub in MODEL_DIRS:
        root = os.path.join(folder_paths.models_dir, sub)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, name, "config.json")):
                found.append("%s/%s" % (sub, name))
    return found or ["<no model found>"]


class LocalVLMChat:
    """
    Single-shot chat call against a local vision-language model loaded through
    HF transformers, bypassing ComfyUI's CLIPLoader entirely.

    The CLIPLoader path only supports the text-encoder architectures ComfyUI
    implements itself (Qwen3-VL 4B/8B, Gemma-3/4 vision), and some checkpoints
    there are conditioning encoders that cannot generate text at all. This node
    loads any HF image-text-to-text model from models/prompt_generator or
    models/LLM instead, which is what makes larger or purpose-built captioners
    (JoyCaption, abliterated Qwen3-VL) usable.
    """

    def __init__(self):
        self.processor = None
        self.model = None
        self.loaded_key = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (discover_models(), {
                    "tooltip": "HF model directory under models/prompt_generator "
                               "or models/LLM. Refresh the browser after adding one.",
                }),
                "user_prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "tooltip": "Folded into the user turn for templates that "
                               "reject a system role (e.g. Gemma 3).",
                }),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 2.0,
                    "step": 0.05, "round": 0.01,
                    "tooltip": "0 = greedy decoding.",
                }),
                "quantization": (["none", "8bit", "4bit"], {
                    "tooltip": "bitsandbytes on-the-fly quantization. Use for "
                               "models that do not fit in VRAM unquantized.",
                }),
                "keep_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep the model in VRAM between runs. Turn off to "
                               "free VRAM for sampling right after the call.",
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "seed": ("INT", {"forceInput": True, "default": 0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attention": (["sdpa", "eager", "flash_attention_2"],),
                "max_pixels": ("INT", {
                    "default": 0, "min": 0, "max": 16777216,
                    "tooltip": "Vision token budget for Qwen-VL style processors "
                               "(0 = model default). Ignored by other processors.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("Response", "Stats",)
    FUNCTION = "generate"
    CATEGORY = "LLM"

    # ------------------------------------------------------------------ runtime

    def load(self, model, quantization, attention, max_pixels):
        import torch
        from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                 BitsAndBytesConfig)

        key = (model, quantization, attention, int(max_pixels))
        if self.loaded_key == key and self.model is not None:
            return

        self.unload()
        checkpoint = os.path.join(folder_paths.models_dir, *model.split("/"))
        if not os.path.isfile(os.path.join(checkpoint, "config.json")):
            raise RuntimeError("LocalVLMChat: no config.json in %s" % checkpoint)

        # A VLM of this size does not co-exist with loaded diffusion models.
        if model_management is not None:
            model_management.unload_all_models()

        try:
            self.processor = AutoProcessor.from_pretrained(
                checkpoint, **({"max_pixels": int(max_pixels)} if max_pixels > 0 else {}))
        except (TypeError, ValueError):
            self.processor = AutoProcessor.from_pretrained(checkpoint)

        quant_config = None
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        device = self.device()
        bf16 = (torch.cuda.is_available()
                and torch.cuda.get_device_capability(device)[0] >= 8)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map="auto", attn_implementation=attention,
            quantization_config=quant_config)
        self.model.eval()
        self.loaded_key = key
        print("LocalVLMChat: loaded %s (%s, %s)" % (model, quantization, attention))

    def unload(self):
        import torch
        self.processor = None
        self.model = None
        self.loaded_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def device():
        return model_management.get_torch_device() if model_management else "cuda"

    @staticmethod
    def to_pil(image):
        if image is None:
            return []
        import numpy as np
        from PIL import Image

        frames = []
        for frame in image:
            array = (frame.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            frames.append(Image.fromarray(array[:, :, :3]))
        return frames

    def build_text(self, system_prompt, user_prompt, image_count):
        """Render the chat template.

        Templates disagree on the message shape, so try them in order of
        expressiveness: modern ones (Qwen3-VL) take content blocks including an
        explicit image block, some (Gemma 3) reject a system role, and
        Llama/Llava-style ones (JoyCaption) take a plain string and inject the
        image tokens themselves.
        """
        system = system_prompt.strip()
        blocks = [{"type": "image"} for _ in range(image_count)]
        blocks.append({"type": "text", "text": user_prompt})

        if system:
            candidates = [
                [{"role": "system", "content": [{"type": "text", "text": system}]},
                 {"role": "user", "content": blocks}],
                [{"role": "user",
                  "content": [{"type": "text", "text": system}] + blocks}],
                [{"role": "system", "content": system},
                 {"role": "user", "content": user_prompt}],
            ]
        else:
            candidates = [
                [{"role": "user", "content": blocks}],
                [{"role": "user", "content": user_prompt}],
            ]

        last_error = None
        for messages in candidates:
            try:
                return self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception as error:
                last_error = error
        raise RuntimeError(
            "LocalVLMChat: this processor accepted none of the chat template "
            "shapes: %s" % last_error)

    def generate(self, model, user_prompt, system_prompt, max_new_tokens,
                 temperature, quantization, keep_loaded, image=None, seed=0,
                 top_p=0.9, attention="sdpa", max_pixels=0):
        import torch

        start = time.time()
        self.load(model, quantization, attention, max_pixels)

        images = self.to_pil(image)
        text = self.build_text(system_prompt, user_prompt, len(images))
        # No padding: a single prompt needs none, and some tokenizers
        # (JoyCaption) ship without a pad token.
        inputs = self.processor(text=[text], images=images or None,
                               return_tensors="pt")
        inputs = inputs.to(self.device())

        do_sample = float(temperature) > 0.0
        torch.manual_seed(int(seed))
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
                top_p=float(top_p) if do_sample else None)

        trimmed = [out[len(inp):] for inp, out
                   in zip(inputs["input_ids"], generated)]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        response = decoded[0].strip() if decoded else ""

        stats = (
            "model=%s, quantization=%s, attention=%s, images=%d, "
            "max_new_tokens=%d, temperature=%s, seed=%d, chars=%d, %.1fs"
            % (model, quantization, attention, len(images), int(max_new_tokens),
               float(temperature), int(seed), len(response), time.time() - start)
        )
        print("LocalVLMChat: %s" % stats)

        if not keep_loaded:
            self.unload()
        return (response, stats)


NODE_CLASS_MAPPINGS = {
    "LocalVLMChat": LocalVLMChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalVLMChat": "Local VLM Chat (transformers)",
}
