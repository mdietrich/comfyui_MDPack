# comfyui_MDPack

Personal collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Nodes

| Node group | File | Purpose |
|---|---|---|
| Quality Gate | `quality_gate.py` | Scores generated images (aesthetic model `sac+logos+ava1-l14-linearMSE`) and gates low-quality outputs |
| Hand/Foot Montage | `hand_foot_montage.py` | Builds montage crops of hands/feet for detail inspection |
| JSON → Text | `json_to_text.py` | Extracts text fields from JSON payloads |
| OpenRouter Chunked Prompts | `openrouter_chunked_prompts.py` | Generates large prompt batches via the OpenRouter API using JSON mode with adaptive chunking, bypassing output-token limits |
| Key-Value Dropdown | `key_value_dropdown.py` | User-defined `key = integer` pairs shown as a dropdown; outputs the selected value and key. Mapping textarea is collapsed by default (Edit mapping toggle) |
| Hosted model API | `atlascloud_api.py` | `Atlas Cloud / WaveSpeed Image` / `… Video`: generate or edit media through the [atlascloud.ai](https://www.atlascloud.ai) or [wavespeed.ai](https://wavespeed.ai) hosted-model APIs (Seedream, Flux, Imagen, Kling, Veo, Wan, ...), selected per node with the `provider` widget. Images come back as an IMAGE batch, videos as decoded frames plus FPS and file path |
| Character Fanout | `character_fanout.py` | Expands one or both characters into four aligned per-image lists (prompt, trigger word, LoRA, output folder), so a single run can cover both characters |
| Provenance-aware save | `media_save.py` | Saves PNG/JPEG/WebP images or MP4 videos with optional IPTC/XMP AI disclosure, ComfyUI prompt/workflow metadata, and optional signed C2PA Content Credentials |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mdietrich/comfyui_MDPack.git
pip install -r comfyui_MDPack/requirements.txt
```

Restart ComfyUI (or use a hot-reload extension such as LG_HotReload).

## Notes

- `weights/` ships the small aesthetic-scoring checkpoint used by Quality Gate.
- The provenance-aware save nodes use Save Image-style numbered filenames and
  subfolders. `filename_prefix` supports `%batch_num%`, `%width%`, `%height%`,
  `%date:yyyy-MM-dd_HH-mm-ss%`, `%year%`, `%month%`, `%day%`, `%hour%`,
  `%minute%`, `%second%`, and scalar `%node.widget%` values from the queued
  prompt. PNG keeps ComfyUI's standard `prompt` and workflow text chunks;
  JPEG/WebP use embedded XMP. Workflow embedding defaults to off because
  workflows can contain API keys, prompts, and local filesystem paths; review a
  workflow for secrets before enabling it on media that will be shared.
- The video node accepts either an existing video path or an IMAGE frame batch
  plus FPS. For an existing path, `remux_all_streams` is the lossless default:
  it copies every compatible stream, global metadata, and chapters into the MP4
  without re-encoding. Choose `transcode_primary_av` when the source codecs are
  not MP4-compatible; it keeps only the primary video and optional audio stream,
  encodes H.264/AAC, and pads odd dimensions to even dimensions. IMAGE frames
  are streamed to FFmpeg one at a time and encoded as H.264. Encoding/remuxing
  always finishes before XMP and C2PA are added.
- `embed_standalone_ai_xmp` controls only the standalone IPTC/XMP Digital Source
  Type field for either fully generated or AI-edited/composite media. It does
  not control the signed claim: when `sign_c2pa` is enabled, the C2PA manifest
  always contains the normative `c2pa.created` action, its Digital Source Type,
  and `c2pa.metadata`, even when standalone XMP is disabled. Video XMP needs
  `exiftool`; video encoding needs `ffmpeg`, and a video path also needs
  `ffprobe`. Executable paths and a shared `tool_timeout_seconds` value for
  FFmpeg, FFprobe, and ExifTool are configurable on the video node. The timeout
  does not apply to in-process C2PA signing.
- C2PA signing is performed directly in the ComfyUI process with the official
  `c2pa-python` SDK installed by `requirements.txt`; no `c2patool` executable is
  needed. Signing is optional and requires user-supplied PEM certificate-chain
  and private-key files plus the matching signature algorithm. No certificates
  or keys are bundled, and private-key contents are never logged. Keep production
  keys protected; a local filesystem key is appropriate only when the ComfyUI
  host and its file permissions meet your security requirements. Signing is the
  final processing step and fails closed: when requested signing fails, the
  unsigned output is removed instead of being returned. A fully generated asset
  receives a first `c2pa.created` action with the IPTC
  `trainedAlgorithmicMedia` source type.
- Existing API prompts remain compatible: obsolete `c2patool_path` inputs are
  accepted and ignored, and the video timeout keeps its original
  `tool_timeout_seconds` name. When an older graph is opened, the web extension
  removes the former c2patool widget value while preserving the timeout and
  workflow-embedding choices in their correct widgets.
- The OpenRouter nodes expect an `OPENROUTER_API_KEY` (or the key configured in the node inputs).
- `provider` picks the hosted API a node talks to: `atlascloud` (the default) or `wavespeed`. It is
  the first widget, directly above the `api_key` it governs, because it decides what every widget
  below it means — which key is valid, which catalogue the model dropdown shows. Switching provider
  means pasting that provider's key and picking a model from its own catalogue.
  `widgets_values` is a positional array, so putting `provider` first shifts every value of a
  workflow saved without it; the web extension detects that on load and re-applies the values under
  the layout they were written with (both the original pre-provider layout and the short-lived one
  that had `provider` last). Workflows saved from now on carry the provider in first position.
- The nodes take the API key from their own `api_key` input; `model` is free text so any model id of
  the selected provider works (copy it from the model page URL, e.g.
  `bytedance/seedream-v5.0-pro/edit` on Atlas Cloud, `wavespeed-ai/flux-2-pro/text-to-image` on
  WaveSpeed). Model-specific fields that are not widgets go into the `extra_params` JSON input,
  which is merged into the request body last. Generation is asynchronous: the nodes submit the job
  and poll the provider's prediction endpoint until it is done, so a long video keeps the ComfyUI
  worker busy until `timeout` is reached.
- Provider differences worth knowing: WaveSpeed names the model in the request URL rather than the
  body, so nothing is sent under a `model` key there, and it rejects body keys its schema does not
  declare (`additionalProperties: false`) — an `extra_params` typo therefore fails the request
  outright instead of being ignored, and the console line naming the unknown key says which one.
  WaveSpeed also deletes uploaded reference media after 7 days.
- Reference images: connect up to ten IMAGE inputs (`image`, `image_2` .. `image_10`, each may be a
  batch) plus `reference_urls` for already hosted images; they are concatenated in that order. The
  receiving field name is normally read from the selected model's own schema (see "Model
  selection" below) — `image_field` is only needed to override that for a model the schema lookup
  doesn't cover, e.g. `images` for the Seedream edit models, `image` for `atlascloud/step1x-edit`,
  `image_url` for most video models. A plural name receives the list, a singular one only the first
  reference.
- `image_transport` chooses between `upload` (via the provider's media-upload endpoint, small
  payload) and `base64` (inline data URI, no upload). Uploads are cached process-wide by image
  content hash *and provider* for `upload_cache_minutes` (default 120, `0` disables), so re-running
  a workflow with unchanged references costs no further uploads — and switching provider never
  reuses a URL the other provider's host serves.
- Video frame decoding uses PyAV (bundled with ComfyUI) and falls back to OpenCV.
- Prompt lists: the image node takes a whole prompt list (e.g. from `CR Prompt List`) in a
  single execution instead of being run once per prompt, and `max_parallel` sets how many of those
  prompts are generated at the provider at the same time (`1` = one after another, the previous
  behaviour). Reference images are uploaded once and shared by every request; the node returns when
  all of them are done, with every image in one IMAGE batch in prompt order. A request that fails
  does not take the batch down — the images that did come back are returned and `Info` lists the
  failed prompts; only a batch where every request failed raises.

### Model selection

`output_type` and `input_type` filter the model list that the `model` dropdown
offers; both are UI-only and never sent to the API. The list comes from the
selected provider's own catalogue — `https://api.atlascloud.ai/api/v1/models`
(no API key required) or `https://api.wavespeed.ai/api/v3/models` (needs the
key from the `api_key` widget, which the browser sends in an
`X-MDPack-Api-Key` header so it never reaches the server's access log). Each
catalogue is cached separately for 24 hours under `<ComfyUI>/temp/mdpack_atlas/`;
`POST /mdpack/atlas/refresh?provider=<name>` forces a refetch. Switching
`provider` reloads the dropdown and selects that catalogue's first matching
model, since the two catalogues share no model ids.

WaveSpeed's `type` labels only sometimes read `input-to-output`; for the rest
(`upscaler`, `lora-support`, `video-extend`, ...) the kinds are derived from
the model id and from the uploader fields the schema declares, so an
`image-upscaler` lands under Image and a `video-upscaler` under Video. A model
whose kind is guessed wrong is still reachable — `model` accepts any id typed
in directly.

Every widget prefixed `param_` is generated from the selected model's own
schema; the values are stored in `params_json` as `{"field_name": value, ...}`.
A `param_` widget left at the schema's own default is *not* written into
`params_json` — it stays out of the request entirely and the model's own
server-side default applies instead. Only a value you actually change (or one
restored from a saved workflow) is sent, even if you later set the widget back
to what looks like the same default value: once touched, it stays in
`params_json`. This keeps a model's own defaults from silently overriding the
node's built-in `width`/`height`/`seed`/etc. widgets the moment a model is
selected. One consequence: the node does not currently enforce a schema's
`required` fields, so a model that declared a required field with no
server-side default of its own would need that field touched at least once
(or set via `extra_params`) to reach the request — no model in the current
catalogue does this, but it is worth knowing if one ever does.
`extra_params` is merged last and therefore still overrides anything set
through the widgets or `params_json`. `params_json` itself can also be edited
by hand — it is just JSON — which is useful for a field the schema-driven
widgets don't expose (e.g. a nested object).

Image inputs follow the schema too. On WaveSpeed the media fields are
recognised from the uploader hint each field carries (`x-ui-component` plus its
`accept` pattern), which is why the ~30 different names its catalogue uses
(`reference_images`, `first_frame_image`, `clothes_images`, ...) all work, and
why an uploader that takes a training `.zip` stays an ordinary text widget
instead of becoming an image slot. A model with an `images` array shows up to
ten slots, a model with `image` plus `last_image` shows exactly those two, and a
pure text-to-image model hides them all. `input_map` optionally pins a specific
image slot to a schema field name (e.g. `{"last_image": 1}` to say "the second
connected image is the last frame"); it is normally filled in automatically by
the web extension as slots are connected, and left empty means "fill the
schema's fields in order." `image_field` stays empty by default, which means
routing is entirely schema-driven; set it only to override the schema for a
model the schema lookup does not (yet) cover — a plural name (e.g. `images`)
then receives the whole reference list, a singular one (e.g. `image`) only the
first reference.

The nodes print diagnostics to the ComfyUI console rather than failing the
run, since a request that is merely suboptimal should still be attempted:

- **Unknown keys** — a `params_json`/`extra_params` key the model's schema
  doesn't declare is still sent (a model may accept more than its schema
  documents), but printed once so a typo doesn't vanish silently.
- **Out-of-enum size** — a derived `width`x`height` size string that isn't
  among the schema's declared enum values is still sent (the declared list
  reads as a suggestion, not a hard constraint), with a note explaining why.
- **Dropped references** — connected reference images that have nowhere to go
  (the schema declares no image field at all, or `input_map` doesn't cover
  every collected reference) are named explicitly rather than silently
  omitted from the request.
- **Possible `input_map` misroute** — `input_map` indices count *connected
  image slots*, while the request-side reference list counts *images*; if an
  earlier slot carries a multi-image batch, every index after it can point at
  the wrong reference. The node warns when this mismatch is detected, since
  neither side alone can see it (the browser has no visibility into tensor
  shapes; the request-builder has no visibility into which slot is which).
