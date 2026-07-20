# comfyui_MDPack

Personal collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Nodes

| Node group | File | Purpose |
|---|---|---|
| Quality Gate | `quality_gate.py` | Scores generated images (aesthetic model `sac+logos+ava1-l14-linearMSE`) and gates low-quality outputs |
| Hand/Foot Montage | `hand_foot_montage.py` | Builds montage crops of hands/feet for detail inspection |
| JSON → Text | `json_to_text.py` | Extracts text fields from JSON payloads |
| OpenRouter Chunked Prompts | `openrouter_chunked_prompts.py` | Generates large prompt batches via the OpenRouter API using JSON mode with adaptive chunking, bypassing output-token limits |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mdietrich/comfyui_MDPack.git
pip install -r comfyui_MDPack/requirements.txt
```

Restart ComfyUI (or use a hot-reload extension such as LG_HotReload).

## Notes

- `weights/` ships the small aesthetic-scoring checkpoint used by Quality Gate.
- The OpenRouter nodes expect an `OPENROUTER_API_KEY` (or the key configured in the node inputs).
