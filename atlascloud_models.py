"""Model catalogues and per-model request schemas for the hosted providers.

Two providers are supported, both of which publish a machine-readable
catalogue of every model they host:

  atlascloud  GET https://api.atlascloud.ai/api/v1/models
              needs no API key, answers with {"code": "200", "data": [...]}.
              Each entry carries the model id, a display name, exactly one
              category of the form INPUT-TO-OUTPUT, and the URL of an OpenAPI
              schema describing the request body that model accepts.

  wavespeed   GET https://api.wavespeed.ai/api/v3/models
              needs a Bearer key and answers with {"code": 200, "data": [...]}.
              Each entry carries model_id, a coarse `type` (text-to-image,
              upscaler, ...) and -- unlike Atlas -- the request schema inline
              under api_schema.api_schemas[0].request_schema, so no second
              request per model is needed.

Everything downstream (the nodes, the web extension) works on the normalised
shapes produced here, which are identical for both providers.

Both the catalogue and the individual schemas are cached on disk so opening a
workflow does not hit the network, and so a failed refresh still yields the
previously known data.
"""

import hashlib
import json
import os
import tempfile
import time

import requests

try:
    import folder_paths
except ImportError:  # allows importing the module outside of ComfyUI
    folder_paths = None


CATALOGUE_URL = "https://api.atlascloud.ai/api/v1/models"
CATALOGUE_CACHE_FILE = "models.json"
DEFAULT_MAX_AGE_SECONDS = 24 * 60 * 60

WAVESPEED_CATALOGUE_URL = "https://api.wavespeed.ai/api/v3/models"

DEFAULT_PROVIDER = "atlascloud"

# Everything that differs between the two providers, in one place. The
# catalogue cache file is per provider so switching the widget never serves
# one provider's models under the other's name.
PROVIDERS = {
    "atlascloud": {
        "label": "Atlas Cloud",
        "catalogue_url": CATALOGUE_URL,
        "catalogue_cache_file": CATALOGUE_CACHE_FILE,
        "catalogue_needs_key": False,
    },
    "wavespeed": {
        "label": "WaveSpeed",
        "catalogue_url": WAVESPEED_CATALOGUE_URL,
        "catalogue_cache_file": "models_wavespeed.json",
        "catalogue_needs_key": True,
    },
}

PROVIDER_IDS = list(PROVIDERS)


def provider_config(provider):
    """The provider table entry, defaulting to Atlas Cloud for an empty name.

    An unknown name is an error rather than a silent fallback: it would
    otherwise send a WaveSpeed model id to Atlas Cloud and fail much later
    with a confusing "model not found".
    """
    key = (provider or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of {', '.join(PROVIDER_IDS)}"
        )
    return key, PROVIDERS[key]


def provider_label(provider):
    return provider_config(provider)[1]["label"]


def cache_directory():
    """Directory holding the catalogue and schema caches; created on demand."""
    if folder_paths is not None:
        base_directory = folder_paths.get_temp_directory()
    else:
        base_directory = tempfile.gettempdir()
    directory = os.path.join(base_directory, "mdpack_atlas")
    os.makedirs(directory, exist_ok=True)
    return directory


def fetch_catalogue(timeout=30.0, provider=DEFAULT_PROVIDER, api_key=""):
    """Download the full model list. Raises on network or format errors."""
    provider, config = provider_config(provider)
    label = config["label"]
    headers = {}
    if config["catalogue_needs_key"]:
        if not (api_key or "").strip():
            raise RuntimeError(
                f"{label}: the model list needs an API key. Paste the key from the "
                f"{label} console into the node's api_key widget."
            )
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    response = requests.get(config["catalogue_url"], headers=headers, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(
            f"{label}: model list failed with HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    payload = response.json()
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(
            f"{label}: model list contained no models: {json.dumps(payload)[:300]}"
        )
    return entries


def _cache_path(provider=DEFAULT_PROVIDER):
    _, config = provider_config(provider)
    return os.path.join(cache_directory(), config["catalogue_cache_file"])


def _read_cache(provider=DEFAULT_PROVIDER):
    try:
        with open(_cache_path(provider), "r", encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
    except (OSError, ValueError):
        return None, 0.0
    if not isinstance(cached, dict):
        return None, 0.0
    entries = cached.get("models")
    if not isinstance(entries, list) or not entries:
        return None, 0.0
    return entries, float(cached.get("fetched_at", 0.0))


def _write_cache(entries, provider=DEFAULT_PROVIDER):
    try:
        with open(_cache_path(provider), "w", encoding="utf-8") as cache_file:
            json.dump({"fetched_at": time.time(), "models": entries}, cache_file)
    except OSError as error:
        print(f"{provider_label(provider)}: could not write the model cache: {error}")


def load_catalogue(max_age_seconds=DEFAULT_MAX_AGE_SECONDS, force_refresh=False, timeout=30.0,
                   provider=DEFAULT_PROVIDER, api_key=""):
    """Return the model list, preferring a fresh disk cache over the network.

    A failing refresh falls back to the cached list (however old), because a
    stale dropdown is far more useful than an empty one. That also means a
    WaveSpeed catalogue downloaded once keeps working while the api_key
    widget is empty (e.g. in the browser before the key is pasted).
    """
    provider, _ = provider_config(provider)
    cached_entries, fetched_at = _read_cache(provider)
    cache_is_fresh = cached_entries is not None and (time.time() - fetched_at) < max_age_seconds
    if cache_is_fresh and not force_refresh:
        return cached_entries
    try:
        entries = fetch_catalogue(timeout=timeout, provider=provider, api_key=api_key)
    except Exception as error:
        if cached_entries is not None:
            print(
                f"{provider_label(provider)}: model list refresh failed ({error}); "
                f"using the cached list"
            )
            return cached_entries
        raise
    _write_cache(entries, provider)
    return entries


OUTPUT_KINDS = ["Image", "Video", "Audio", "3D", "Text"]
INPUT_KINDS = ["Text", "Image", "Video", "Audio"]

# Atlas categories name the modality on both sides of "-TO-", with a few
# wordings that do not match the kind names used in the dropdowns.
_KIND_ALIASES = {
    "IMAGE": "Image",
    "TEXT": "Text",
    "VIDEO": "Video",
    "AUDIO": "Audio",
    "SPEECH": "Audio",
    "3D": "3D",
    "REFERENCE": "Image",  # a "reference" input is one or more reference images
}

# Categories that do not follow the INPUT-TO-OUTPUT pattern.
_SPECIAL_CATEGORIES = {
    "IMAGE-TOOLS": ("Image", "Image"),
    "LLM": ("Text", "Text"),
}


def split_category(category):
    """Turn a category such as IMAGE-TO-VIDEO into ("Image", "Video")."""
    key = (category or "").strip().upper()
    if key in _SPECIAL_CATEGORIES:
        return _SPECIAL_CATEGORIES[key]
    if "-TO-" not in key:
        return ("", "")
    input_part, output_part = key.split("-TO-", 1)
    return (_KIND_ALIASES.get(input_part, ""), _KIND_ALIASES.get(output_part, ""))


def _category_from_tags(tags):
    """Recover a category from a tag, for entries whose `categories` is null.

    Atlas encodes the same information as a tag in that case, but with
    inconsistent punctuation (hyphens or underscores) and stray whitespace.
    """
    for tag in tags or []:
        candidate = (tag or "").strip().replace("_", "-").upper()
        if "-TO-" in candidate or candidate == "LLM":
            return candidate
    return ""


# WaveSpeed's `type` is a single coarse label. Roughly three quarters of them
# follow the same INPUT-TO-OUTPUT wording Atlas uses and go through
# split_category; the rest name a task instead of a modality, so the output
# kind has to be recovered from the type and the model id.
#
# Only the types whose output is *not* readable from the model id are listed
# here -- a trainer returns a LoRA and a moderator returns a verdict, no
# matter what the id says. Everything else (upscaler, ai-remover,
# portrait-transfer, lora-support, video-dubbing) covers both image and video
# variants under one type and is decided by the id instead.
_WAVESPEED_TYPE_OUTPUT = {
    "training": "Text",
    "content-moderation": "Text",
    "llm": "Text",
    "digital-human": "Video",
    "video-extend": "Video",
    "video-effects": "Video",
    "motion-control": "Video",
}

# Keyword -> kind, matched against the model id. The *last* match in the id
# wins, because a model id reads input-to-output just like a category does:
# "kling-video-to-audio" produces audio, "p-image/upscale" an image.
_WAVESPEED_ID_KEYWORDS = (
    ("image", "Image"), ("photo", "Image"), ("portrait", "Image"),
    ("video", "Video"), ("audio", "Audio"), ("sfx", "Audio"), ("speech", "Audio"),
    ("voice", "Audio"), ("music", "Audio"), ("tts", "Audio"), ("3d", "3D"),
)


def _kind_from_identifier(model_identifier):
    """The kind the last modality keyword in a model id names, if any."""
    text = (model_identifier or "").lower()
    best_position, best_kind = -1, ""
    for keyword, kind in _WAVESPEED_ID_KEYWORDS:
        position = text.rfind(keyword)
        if position > best_position:
            best_position, best_kind = position, kind
    return best_kind


def _wavespeed_request_schema(raw):
    """The inline request schema WaveSpeed ships with every catalogue entry."""
    api_schemas = ((raw or {}).get("api_schema") or {}).get("api_schemas") or []
    for entry in api_schemas:
        if (entry or {}).get("type", "model_run") == "model_run":
            return entry.get("request_schema") or {}, entry.get("api_path") or ""
    return {}, ""


def _wavespeed_input_kind(raw):
    """What a model takes, read from the uploader fields its schema declares."""
    request_schema, _ = _wavespeed_request_schema(raw)
    kinds = set()
    for name, definition in (request_schema.get("properties") or {}).items():
        kind = _wavespeed_media_kind(name, definition or {})
        if kind:
            kinds.add(kind)
    for kind in ("Image", "Video", "Audio"):
        if kind.lower() in kinds:
            return kind
    return "Text"


def _normalise_wavespeed_entry(raw):
    model_identifier = raw.get("model_id") or raw.get("name") or ""
    category = raw.get("type") or ""
    input_kind, output_kind = split_category(category)
    if not output_kind:
        output_kind = _WAVESPEED_TYPE_OUTPUT.get(category.strip().lower(), "")
    if not output_kind:
        output_kind = _kind_from_identifier(model_identifier) or "Image"
    if not input_kind:
        input_kind = _wavespeed_input_kind(raw)
    base_price = raw.get("base_price")
    return {
        "model": model_identifier,
        "display_name": raw.get("name") or model_identifier,
        "category": category,
        "input_kind": input_kind,
        "output_kind": output_kind,
        # WaveSpeed ships the schema inline, so there is nothing to fetch
        # later; schema_for_model reads it straight out of the catalogue.
        "schema_url": "",
        "price": "" if base_price is None else str(base_price),
    }


def normalise_entry(raw, provider=DEFAULT_PROVIDER):
    """Reduce a catalogue entry to the fields the nodes and the UI need."""
    provider, _ = provider_config(provider)
    if provider == "wavespeed":
        return _normalise_wavespeed_entry(raw)
    categories = raw.get("categories") or []
    if categories:
        category = categories[0]
        input_kind, output_kind = split_category(category)
    else:
        # No categories at all: try a tag that encodes one, then fall back
        # to the coarse `type` field (output kind only, input stays unknown)
        # so the model is at least reachable rather than silently dropped.
        category = _category_from_tags(raw.get("tags"))
        input_kind, output_kind = split_category(category)
        if not output_kind:
            output_kind = _KIND_ALIASES.get((raw.get("type") or "").strip().upper(), "")
            if output_kind:
                input_kind = ""
                category = raw.get("type", "")
    price = ""
    price_block = raw.get("price") or {}
    actual = price_block.get("actual") or {}
    if isinstance(actual, dict):
        price = str(actual.get("base_price", "") or "")
    return {
        "model": raw.get("model", ""),
        "display_name": raw.get("displayName") or raw.get("model", ""),
        "category": category,
        "input_kind": input_kind,
        "output_kind": output_kind,
        "schema_url": raw.get("schema", ""),
        "price": price,
    }


def list_models(output_kind="", input_kind="", provider=DEFAULT_PROVIDER, **load_kwargs):
    """Normalised catalogue filtered by kind; an empty kind matches anything."""
    provider, _ = provider_config(provider)
    entries = [
        normalise_entry(raw, provider)
        for raw in load_catalogue(provider=provider, **load_kwargs)
    ]
    entries = [entry for entry in entries if entry["model"]]
    if output_kind:
        entries = [entry for entry in entries if entry["output_kind"] == output_kind]
    if input_kind:
        entries = [entry for entry in entries if entry["input_kind"] == input_kind]
    return sorted(entries, key=lambda entry: entry["display_name"].lower())


# Names Atlas uses for reference media inside a request body. A field only
# counts as media when it also has a media-ish type (string URL or list of
# string URLs), so a "video_prompt" text field is not mistaken for an upload.
# Split by kind: both Atlas Cloud nodes only ever carry IMAGE inputs, so a
# video/audio field must never be offered as an image slot or filled with an
# uploaded image reference the way an image-kind field is (see
# image_kind_fields below and the final review, Finding 4).
IMAGE_FIELD_NAMES = {
    "image", "images", "image_url", "image_urls", "input_image", "input_images",
    "first_image", "last_image", "end_image", "start_image", "reference_image",
    "reference_images", "subject_image", "mask", "mask_url",
}
VIDEO_FIELD_NAMES = {"video", "video_url"}
AUDIO_FIELD_NAMES = {"audio", "audio_url"}
MEDIA_FIELD_NAMES = IMAGE_FIELD_NAMES | VIDEO_FIELD_NAMES | AUDIO_FIELD_NAMES


def _media_kind(name):
    if name in IMAGE_FIELD_NAMES:
        return "image"
    if name in VIDEO_FIELD_NAMES:
        return "video"
    if name in AUDIO_FIELD_NAMES:
        return "audio"
    return None  # unreachable for a name already confirmed to be in MEDIA_FIELD_NAMES


# WaveSpeed marks every field that takes a hosted media URL with a UI hint,
# which is a far better signal than a hand-maintained name list: the live
# catalogue uses ~30 different names for image fields alone (image, images,
# reference_images, first_frame_image, clothes_images, texture_image, ...).
_WAVESPEED_UPLOAD_COMPONENTS = {"uploader", "uploaders", "upload", "upload-array", "mask"}


def _wavespeed_media_kind(name, definition):
    """The media kind a WaveSpeed field takes, or None when it takes none.

    A field counts as media when it is marked as an uploader (or carries one
    of the names Atlas already knows) *and* holds URL strings. The kind comes
    from the declared accept pattern where there is one; a non-media accept
    (a training .zip, an .srt subtitle) deliberately yields None so the field
    stays an ordinary text widget the user can paste a URL into, rather than
    becoming an image slot that can never be filled.
    """
    definition = definition or {}
    if definition.get("type") == "array":
        if (definition.get("items") or {}).get("type") != "string":
            return None
    elif definition.get("type") != "string":
        return None
    component = definition.get("x-ui-component")
    if component not in _WAVESPEED_UPLOAD_COMPONENTS and name not in MEDIA_FIELD_NAMES:
        return None
    accept = (definition.get("x-ui-component-props") or {}).get("accept")
    accept = (accept or definition.get("x-accept") or "").strip().lower()
    if accept:
        for prefix in ("image", "video", "audio"):
            if accept.startswith(prefix):
                return prefix
        return None
    kind = _media_kind(name)
    if kind:
        return kind
    identified = _kind_from_identifier(name)
    return identified.lower() if identified in ("Image", "Video", "Audio") else "image"

# Fields the node fills itself; never offered as a schema-driven widget.
_NODE_OWNED_FIELDS = {"model", "prompt", "enable_base64_output"}

_schema_memo = {}


def clear_schema_memo():
    """Drop the in-process schema cache (used by tests and the refresh route)."""
    _schema_memo.clear()


def _schema_cache_path(schema_url):
    digest = hashlib.sha256(schema_url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_directory(), f"schema_{digest}.json")


def fetch_schema(schema_url, timeout=30.0):
    """Download a model schema, falling back to the disk cache on failure."""
    cache_path = _schema_cache_path(schema_url)
    try:
        response = requests.get(schema_url, timeout=timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}")
        document = response.json()
    except Exception as error:
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                print(f"AtlasCloud: schema refresh failed ({error}); using the cached schema")
                return json.load(cache_file)
        except (OSError, ValueError):
            raise RuntimeError(f"Atlas Cloud: could not load schema {schema_url}: {error}")
    # A failure writing the cache must not discard an otherwise successful
    # download, and must not be reported as a fetch failure.
    try:
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump(document, cache_file)
    except OSError as error:
        print(f"AtlasCloud: could not write the schema cache: {error}")
    return document


def _input_properties(document):
    schemas = ((document or {}).get("components") or {}).get("schemas") or {}
    input_schema = schemas.get("Input") or {}
    properties = input_schema.get("properties") or {}
    required = input_schema.get("required") or []
    order = input_schema.get("x-order-properties") or list(properties.keys())
    # x-order-properties is Atlas's curated display order; a property missing
    # from it is an edge case and is appended after the curated ones so it
    # cannot push an important field (e.g. size, duration) further down.
    ordered_names = [name for name in order if name in properties]
    ordered_names += [name for name in properties if name not in ordered_names]
    return properties, set(required), ordered_names


def _wavespeed_input_properties(document):
    """The same (properties, required, order) triple for a WaveSpeed entry.

    `document` is the raw catalogue entry, since WaveSpeed carries the request
    schema inline instead of behind a second URL.
    """
    request_schema, _ = _wavespeed_request_schema(document)
    properties = request_schema.get("properties") or {}
    required = request_schema.get("required") or []
    order = request_schema.get("x-order-properties") or list(properties.keys())
    ordered_names = [name for name in order if name in properties]
    ordered_names += [name for name in properties if name not in ordered_names]
    return properties, set(required), ordered_names


def _is_media_field(name, definition):
    if name not in MEDIA_FIELD_NAMES:
        return False
    if definition.get("type") == "array":
        return (definition.get("items") or {}).get("type") == "string"
    return definition.get("type") == "string"


def normalise_schema(model_identifier, document, provider=DEFAULT_PROVIDER):
    """Reduce a provider's request schema to the fields the node needs."""
    provider, _ = provider_config(provider)
    is_wavespeed = provider == "wavespeed"
    if is_wavespeed:
        properties, required, ordered_names = _wavespeed_input_properties(document)
    else:
        properties, required, ordered_names = _input_properties(document)
    fields = []
    image_fields = []
    prompt_field = "prompt" if "prompt" in properties else ""
    for name in ordered_names:
        definition = properties[name] or {}
        media_kind = (
            _wavespeed_media_kind(name, definition) if is_wavespeed
            else (_media_kind(name) if _is_media_field(name, definition) else None)
        )
        if media_kind:
            is_list = definition.get("type") == "array"
            image_fields.append({
                "name": name,
                "is_list": is_list,
                "max_items": int(definition.get("maxItems", 10)) if is_list else 1,
                "kind": media_kind,
            })
            continue
        if name in _NODE_OWNED_FIELDS:
            continue
        # WaveSpeed marks the fields it drives itself (enable_sync_mode,
        # enable_base64_output) as disabled; offering them as widgets would
        # only invite a request the API rejects.
        if is_wavespeed and definition.get("disabled") is True:
            continue
        fields.append({
            "name": name,
            "type": definition.get("type", "string"),
            "default": definition.get("default"),
            "enum": definition.get("enum"),
            "required": name in required,
            "description": definition.get("description", ""),
            "is_media": False,
            "is_list": definition.get("type") == "array",
        })
    return {
        "model": model_identifier,
        "prompt_field": prompt_field,
        "image_fields": image_fields,
        "fields": fields,
    }


def _wavespeed_schema(model_identifier, force_refresh=False, timeout=30.0, api_key=""):
    for raw in load_catalogue(
        force_refresh=force_refresh, timeout=timeout, provider="wavespeed", api_key=api_key
    ):
        if (raw.get("model_id") or raw.get("name")) == model_identifier:
            return normalise_schema(model_identifier, raw, provider="wavespeed")
    return None


def schema_for_model(model_identifier, force_refresh=False, timeout=30.0,
                     provider=DEFAULT_PROVIDER, api_key=""):
    """Normalised schema for a model id, memoised for the process lifetime.

    The memo is keyed by provider as well: both catalogues host models under
    the same vendor prefixes (bytedance/..., kwaivgi/...) and the very same id
    can exist on both with different fields.
    """
    provider, _ = provider_config(provider)
    memo_key = (provider, model_identifier)
    if not force_refresh and memo_key in _schema_memo:
        return _schema_memo[memo_key]
    empty = {"model": model_identifier, "prompt_field": "", "image_fields": [], "fields": []}
    if provider == "wavespeed":
        normalised = _wavespeed_schema(
            model_identifier, force_refresh=force_refresh, timeout=timeout, api_key=api_key
        )
        if normalised is None:
            return empty
        _schema_memo[memo_key] = normalised
        return normalised
    schema_url = ""
    for entry in list_models():
        if entry["model"] == model_identifier:
            schema_url = entry["schema_url"]
            break
    if not schema_url:
        return empty
    normalised = normalise_schema(model_identifier, fetch_schema(schema_url, timeout=timeout))
    _schema_memo[memo_key] = normalised
    return normalised


def submit_path(model_identifier, provider=DEFAULT_PROVIDER, api_key=""):
    """The api_path WaveSpeed declares for a model, or "" when unknown.

    WaveSpeed puts the model id in the URL instead of the request body. The
    path is almost always "/api/v3/<model id>", but it is taken from the
    schema where the catalogue offers one rather than assembled by hand.
    """
    provider, _ = provider_config(provider)
    if provider != "wavespeed":
        return ""
    try:
        entries = load_catalogue(provider="wavespeed", api_key=api_key)
    except Exception:
        return ""
    for raw in entries:
        if (raw.get("model_id") or raw.get("name")) == model_identifier:
            return _wavespeed_request_schema(raw)[1]
    return ""


def _field_by_name(schema, name):
    for field in schema.get("fields") or []:
        if field["name"] == name:
            return field
    return None


def image_kind_fields(schema):
    """The schema's media fields that accept images, excluding any that only
    take a video or audio reference. Both Atlas Cloud nodes only ever carry
    IMAGE inputs, so a video/audio field cannot be filled from them -- see
    the final review, Finding 4. `kind` defaults to "image" for schemas built
    before this field existed (the two hand-written fallback schemas in
    atlascloud_api.py), so those keep working unchanged."""
    return [field for field in (schema.get("image_fields") or [])
            if field.get("kind", "image") == "image"]


# The names the live catalogue uses for "how many images should this return",
# most common first. Of the 119 image models Atlas Cloud serves, 20 declare
# "num_images", 7 "n" (wan-2.7, gpt-image-1, ERNIE) and 6 "max_images" (the
# Seedream /sequential variants); the remaining ~86 -- among them
# bytedance/seedream-v5.0-pro/edit -- declare none at all and always return a
# single image. A node widget named num_images therefore cannot be sent under
# that name unconditionally: the API silently ignores the unknown key and the
# user gets one image without being told why.
IMAGE_COUNT_FIELD_NAMES = ("num_images", "n", "max_images", "num_outputs")


def image_count_field(schema):
    """The name this model takes for the number of images to return, or None
    when it declares no such field (and hence always returns exactly one)."""
    for name in IMAGE_COUNT_FIELD_NAMES:
        if _field_by_name(schema or {}, name) is not None:
            return name
    return None


def build_payload(model_identifier, prompt, parameters, width=0, height=0, schema=None,
                  provider=DEFAULT_PROVIDER):
    """Assemble a request body using the field names the model actually takes."""
    provider, _ = provider_config(provider)
    schema = schema if schema is not None else schema_for_model(model_identifier, provider=provider)
    # WaveSpeed identifies the model through the request URL and rejects
    # unknown body keys outright (additionalProperties: false), so "model"
    # may only be sent to Atlas Cloud.
    payload = {} if provider == "wavespeed" else {"model": model_identifier}
    label = provider_label(provider)
    prompt_field = schema.get("prompt_field") or ""
    if prompt and prompt_field:
        payload[prompt_field] = prompt
    elif prompt:
        print(
            f"{label}: {model_identifier} declares no prompt field; "
            f"dropping the supplied prompt"
        )

    unknown_keys = []
    for name, value in (parameters or {}).items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        payload[name] = value
        # A model may accept more than its schema documents, so unknown keys
        # are still sent -- just flagged once so a typo does not vanish silently.
        if _field_by_name(schema, name) is None:
            unknown_keys.append(name)
    if unknown_keys:
        print(
            f"{label}: {model_identifier} does not declare {unknown_keys} in its schema; "
            f"sending anyway"
        )

    width_value, height_value = int(width or 0), int(height or 0)
    if width_value > 0 and height_value > 0:
        size_field = _field_by_name(schema, "size")
        if size_field is not None and size_field["type"] == "string":
            if "size" in payload:
                # `parameters` (params_json/extra_params) already decided the
                # size -- explicit precedence is kept, but silently dropping
                # the widget's width/height is exactly the failure mode this
                # feature exists to remove, so it is named instead.
                print(
                    f"{label}: {model_identifier} already has size={payload['size']!r} "
                    f"from parameters; ignoring node width={width_value} height={height_value}"
                )
            else:
                derived_size = f"{width_value}*{height_value}"
                enum_values = size_field.get("enum")
                if enum_values and derived_size not in enum_values:
                    # The declared values read as a suggestion list, not a hard
                    # constraint (Atlas's own description documents free sizes
                    # within pixel/aspect limits), so we still send the derived
                    # size rather than guess-snapping to a nearby enum member.
                    print(
                        f"{label}: {model_identifier} derived size {derived_size!r} is not "
                        f"among the declared values {enum_values}; sending it anyway"
                    )
                payload["size"] = derived_size
        elif _field_by_name(schema, "width") is not None:
            if "width" in payload or "height" in payload:
                print(
                    f"{label}: {model_identifier} already has width={payload.get('width')!r} "
                    f"height={payload.get('height')!r} from parameters; ignoring node "
                    f"width={width_value} height={height_value}"
                )
            payload.setdefault("width", width_value)
            payload.setdefault("height", height_value)
        else:
            print(
                f"{label}: {model_identifier} takes neither size nor width/height; "
                f"ignoring {width_value}x{height_value}"
            )
    return payload


def assign_media(payload, schema, media_urls, input_map=None,
                  model_identifier="", label="AtlasCloud"):
    """Write reference URLs into the image-kind media fields the schema
    declares (see image_kind_fields -- a video/audio field is never filled
    from here, since both Atlas Cloud nodes only ever collect IMAGE inputs).

    `input_map` optionally pins a field to a specific reference index, which is
    how the UI lets a user decide which connected image is the last frame.
    Without it, list fields take every URL and singular fields are filled in
    schema order.
    """
    media_fields = image_kind_fields(schema)
    if not media_fields or not media_urls:
        return payload
    if input_map:
        for field_name, index in input_map.items():
            if 0 <= int(index) < len(media_urls):
                field = next((f for f in media_fields if f["name"] == field_name), None)
                if field is None:
                    continue
                payload[field_name] = (
                    [media_urls[int(index)]] if field["is_list"] else media_urls[int(index)]
                )
        return payload
    remaining = list(media_urls)
    for field in media_fields:
        if not remaining:
            break
        if field["is_list"]:
            if len(remaining) > field["max_items"]:
                print(
                    f"{label}: {model_identifier!r} field '{field['name']}' accepts at most "
                    f"{field['max_items']} reference(s); sending {field['max_items']} of the "
                    f"{len(remaining)} supplied"
                )
            payload[field["name"]] = remaining[: field["max_items"]]
            remaining = remaining[field["max_items"]:]
        else:
            payload[field["name"]] = remaining.pop(0)
    return payload
