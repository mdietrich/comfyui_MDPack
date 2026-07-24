"""MDPack — eigene ComfyUI-Nodes (Quality Gate, Hand/Foot-Montage, JSON→Text, OpenRouter Chunked Prompts, Key-Value Dropdown)."""

from .quality_gate import NODE_CLASS_MAPPINGS as _N1, NODE_DISPLAY_NAME_MAPPINGS as _D1
from .hand_foot_montage import NODE_CLASS_MAPPINGS as _N2, NODE_DISPLAY_NAME_MAPPINGS as _D2
from .json_to_text import NODE_CLASS_MAPPINGS as _N3, NODE_DISPLAY_NAME_MAPPINGS as _D3
from .openrouter_chunked_prompts import NODE_CLASS_MAPPINGS as _N4, NODE_DISPLAY_NAME_MAPPINGS as _D4
from .key_value_dropdown import NODE_CLASS_MAPPINGS as _N5, NODE_DISPLAY_NAME_MAPPINGS as _D5

NODE_CLASS_MAPPINGS = {**_N1, **_N2, **_N3, **_N4, **_N5}
NODE_DISPLAY_NAME_MAPPINGS = {**_D1, **_D2, **_D3, **_D4, **_D5}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
