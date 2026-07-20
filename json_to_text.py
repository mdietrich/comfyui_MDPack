import json


def _decode_escapes(s):
    # interpret escape sequences typed literally into the widget (\n, \t, \r)
    return (s.replace("\\r\\n", "\n")
             .replace("\\n", "\n")
             .replace("\\t", "\t")
             .replace("\\r", "\r"))


def _strip_fences(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


class JSONToText:
    """
    Parse a JSON string (e.g. the OpenRouter node output) and join its values
    into a single string separated by `separator`.

    - JSON array  -> its elements are joined.
    - JSON object -> if `key` is set, that value is used; otherwise a single
      list-valued field is auto-detected, else all values are joined.
    Non-string elements are serialized back to compact JSON.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True}),
                "separator": ("STRING", {"default": "\\n", "multiline": False}),
            },
            "optional": {
                "key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "convert"
    CATEGORY = "text"

    def convert(self, json_string, separator, key=""):
        try:
            data = json.loads(_strip_fences(json_string))
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONToText: input is not valid JSON ({e}).")

        if isinstance(data, dict):
            key = key.strip()
            if key:
                if key not in data:
                    raise ValueError(f"JSONToText: key '{key}' not found in {list(data.keys())}.")
                data = data[key]
            else:
                list_values = [v for v in data.values() if isinstance(v, list)]
                data = list_values[0] if len(list_values) == 1 else list(data.values())

        if not isinstance(data, list):
            data = [data]

        parts = [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                 for item in data]
        return (_decode_escapes(separator).join(parts),)


NODE_CLASS_MAPPINGS = {"JSONToText": JSONToText}
NODE_DISPLAY_NAME_MAPPINGS = {"JSONToText": "JSON → Text (join)"}
