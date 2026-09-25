"""MDPack — eigene ComfyUI-Nodes (Quality Gate, Hand/Foot-Montage, JSON→Text, OpenRouter Chunked Prompts, Key-Value Dropdown, Atlas Cloud API, Character Fanout, Provenance Save, Local Chunked Prompts, Resolution Selector, Video to H3 Clip Prompts)."""

from .quality_gate import NODE_CLASS_MAPPINGS as _N1, NODE_DISPLAY_NAME_MAPPINGS as _D1
from .hand_foot_montage import NODE_CLASS_MAPPINGS as _N2, NODE_DISPLAY_NAME_MAPPINGS as _D2
from .json_to_text import NODE_CLASS_MAPPINGS as _N3, NODE_DISPLAY_NAME_MAPPINGS as _D3
from .openrouter_chunked_prompts import NODE_CLASS_MAPPINGS as _N4, NODE_DISPLAY_NAME_MAPPINGS as _D4
from .key_value_dropdown import NODE_CLASS_MAPPINGS as _N5, NODE_DISPLAY_NAME_MAPPINGS as _D5
from .atlascloud_api import NODE_CLASS_MAPPINGS as _N6, NODE_DISPLAY_NAME_MAPPINGS as _D6
from .character_fanout import NODE_CLASS_MAPPINGS as _N7, NODE_DISPLAY_NAME_MAPPINGS as _D7
from .media_save import NODE_CLASS_MAPPINGS as _N8, NODE_DISPLAY_NAME_MAPPINGS as _D8
from .local_chunked_prompts import NODE_CLASS_MAPPINGS as _N9, NODE_DISPLAY_NAME_MAPPINGS as _D9
from .resolution_selector import NODE_CLASS_MAPPINGS as _N10, NODE_DISPLAY_NAME_MAPPINGS as _D10
from .video_to_h3_prompts import NODE_CLASS_MAPPINGS as _N11, NODE_DISPLAY_NAME_MAPPINGS as _D11
from .local_llm_chat import NODE_CLASS_MAPPINGS as _N12, NODE_DISPLAY_NAME_MAPPINGS as _D12
from .local_vlm_chat import NODE_CLASS_MAPPINGS as _N13, NODE_DISPLAY_NAME_MAPPINGS as _D13
from . import atlascloud_routes  # noqa: F401  (registers the Atlas Cloud routes)

NODE_CLASS_MAPPINGS = {**_N1, **_N2, **_N3, **_N4, **_N5, **_N6, **_N7, **_N8, **_N9, **_N10, **_N11, **_N12, **_N13}
NODE_DISPLAY_NAME_MAPPINGS = {**_D1, **_D2, **_D3, **_D4, **_D5, **_D6, **_D7, **_D8, **_D9, **_D10, **_D11, **_D12, **_D13}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
