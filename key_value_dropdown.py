class KeyValueDropdown:
    """
    User-facing dropdown over freely definable key/value pairs.

    The `mapping` text defines one `key = integer` pair per line (lines
    starting with # and blank lines are ignored). The accompanying frontend
    extension (web/keyValueDropdown.js) renders `selection` as a dropdown of
    the mapping keys and mirrors `mapping` into a node property so the pairs
    can also be edited via the properties panel.

    Outputs: the integer assigned to the selected key, the key itself, and the
    raw mapping text (so downstream nodes can resolve the other keys).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mapping": ("STRING", {
                    "multiline": True,
                    "default": "Low = 1\nMedium = 2\nHigh = 3",
                    "tooltip": "One 'key = integer' pair per line. "
                               "Lines starting with # are ignored.",
                }),
                "selection": (["Low", "Medium", "High"], {
                    "default": "Low",
                    "tooltip": "The selected key; shown as a dropdown of the "
                               "mapping keys.",
                }),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, selection):
        # The dropdown options are rebuilt dynamically in the frontend from
        # the mapping text, so the static combo list above must not be
        # enforced; select() validates against the actual mapping.
        return True

    RETURN_TYPES = ("INT", "STRING", "STRING")
    RETURN_NAMES = ("value", "key", "mapping")
    FUNCTION = "select"
    CATEGORY = "MDPack"

    @staticmethod
    def parse_mapping(mapping):
        pairs = {}
        for raw_line in (mapping or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not sep or not key:
                raise ValueError(
                    f"KeyValueDropdown: invalid mapping line {raw_line!r} "
                    "(expected 'key = integer')"
                )
            try:
                int_value = int(value)
            except ValueError:
                raise ValueError(
                    f"KeyValueDropdown: value for key '{key}' is not an "
                    f"integer: {value!r}"
                )
            if key in pairs:
                raise ValueError(
                    f"KeyValueDropdown: duplicate key '{key}' in mapping"
                )
            pairs[key] = int_value
        if not pairs:
            raise ValueError("KeyValueDropdown: mapping contains no pairs")
        return pairs

    def select(self, mapping, selection):
        pairs = self.parse_mapping(mapping)
        key = (selection or "").strip()
        if key not in pairs:
            raise ValueError(
                f"KeyValueDropdown: selection '{key}' not found in mapping "
                f"(available: {', '.join(pairs)})"
            )
        return (pairs[key], key, mapping)


NODE_CLASS_MAPPINGS = {
    "KeyValueDropdown": KeyValueDropdown,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeyValueDropdown": "Key-Value Dropdown",
}
