try:
    from .key_value_dropdown import KeyValueDropdown
except ImportError:  # standalone import from the test scripts
    from key_value_dropdown import KeyValueDropdown


class AnyType(str):
    """Type sentinel that never reports a type mismatch to ComfyUI."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class CharacterFanout:
    """
    Expands one or both characters into aligned per-image lists.

    ComfyUI executes downstream nodes once per element of a list output, so
    emitting four lists of equal length is what makes a single run produce
    images for both characters while every image keeps its own trigger word,
    LoRA and output folder.

    `mode` comes from the Character Selector (KeyValueDropdown): 0 selects the
    character mapped to 0, 1 the character mapped to 1, and 2 runs both — the
    complete block of character 0 first, then the block of character 1. Block
    order (rather than interleaving) keeps the number of LoRA reloads at one.

    In mode 0 and 1 the upstream prompt pipeline already produced the prompts
    for the selected character, so only `prompts_primary` is used.
    """

    MODE_FIRST = 0
    MODE_SECOND = 1
    MODE_BOTH = 2

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ("INT", {
                    "forceInput": True,
                    "tooltip": "0 = first character, 1 = second character, "
                               "2 = both (doubles the image count).",
                }),
                "mapping": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Raw mapping text of the Character Selector; "
                               "supplies the folder name of each character.",
                }),
                "prompts_primary": ("STRING", {"forceInput": True}),
                "prompts_secondary": ("STRING", {"forceInput": True}),
                "trigger_a": ("STRING", {"forceInput": True}),
                "trigger_b": ("STRING", {"forceInput": True}),
                "lora_a": (ANY, {"forceInput": True}),
                "lora_b": (ANY, {"forceInput": True}),
            }
        }

    # lora_name stays wildcard-typed: it feeds a COMBO input
    # (LoraLoaderModelOnly.lora_name), which a plain STRING output cannot
    # connect to in the frontend.
    RETURN_TYPES = ("STRING", "STRING", ANY, "STRING")
    RETURN_NAMES = ("prompt", "triggerword", "lora_name", "character_key")
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION = "fan_out"
    CATEGORY = "MDPack"

    @staticmethod
    def split_prompts(text, field):
        lines = [line.strip() for line in (text or "").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            raise ValueError(
                f"CharacterFanout: {field} contains no prompt lines"
            )
        return lines

    @staticmethod
    def key_for_value(pairs, value):
        for key, mapped in pairs.items():
            if mapped == value:
                return key
        raise ValueError(
            f"CharacterFanout: mapping has no key for value {value} "
            f"(mapping: {', '.join(f'{k}={v}' for k, v in pairs.items())})"
        )

    def fan_out(self, mode, mapping, prompts_primary, prompts_secondary,
                trigger_a, trigger_b, lora_a, lora_b):
        mode = int(mode)
        if mode not in (self.MODE_FIRST, self.MODE_SECOND, self.MODE_BOTH):
            raise ValueError(
                f"CharacterFanout: unsupported mode {mode} "
                "(expected 0, 1 or 2)"
            )

        pairs = KeyValueDropdown.parse_mapping(mapping)
        key_a = self.key_for_value(pairs, 0)
        key_b = self.key_for_value(pairs, 1)

        primary = self.split_prompts(prompts_primary, "prompts_primary")

        if mode == self.MODE_FIRST:
            blocks = [(primary, trigger_a, lora_a, key_a)]
        elif mode == self.MODE_SECOND:
            blocks = [(primary, trigger_b, lora_b, key_b)]
        else:
            secondary = self.split_prompts(prompts_secondary,
                                           "prompts_secondary")
            blocks = [
                (primary, trigger_a, lora_a, key_a),
                (secondary, trigger_b, lora_b, key_b),
            ]

        prompts, triggers, loras, keys = [], [], [], []
        for lines, trigger, lora, key in blocks:
            for line in lines:
                prompts.append(line)
                triggers.append(trigger)
                loras.append(lora)
                keys.append(key)
        return (prompts, triggers, loras, keys)


NODE_CLASS_MAPPINGS = {
    "CharacterFanout": CharacterFanout,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterFanout": "Character Fanout",
}
