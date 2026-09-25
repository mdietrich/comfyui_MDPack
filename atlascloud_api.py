"""Hosted image and video generation nodes for Atlas Cloud and WaveSpeed.

Atlas Cloud (atlascloud.ai) exposes ~600 hosted models (Seedream, Flux,
Imagen, Kling, Veo, Wan, ...) behind three endpoints:

    POST /api/v1/model/generateImage      -> {"data": {"id": "<prediction>"}}
    POST /api/v1/model/generateVideo      -> {"data": {"id": "<prediction>"}}
    POST /api/v1/model/uploadMedia        -> {"url": "https://..."}
    GET  /api/v1/model/prediction/{id}    -> {"data": {"status": ..., "outputs": [...]}}

WaveSpeed (wavespeed.ai) serves ~950 models through the same request/poll
shape, with three differences the `provider` widget switches between:

    POST /api/v3/{model id}               -> {"data": {"id": ..., "urls": {"get": ...}}}
        the model is named by the URL, not by a "model" key in the body, and
        unknown body keys are rejected (additionalProperties: false)
    POST /api/v3/media/upload/binary      -> {"data": {"download_url": "https://..."}}
    GET  /api/v3/predictions/{id}/result  -> {"data": {"status": ..., "outputs": [...]}}

Generation is always asynchronous on both: submit, then poll the prediction
endpoint until the status is terminal. The response shapes differ slightly
between the documented examples per model family, so every extractor here
accepts all documented variants instead of a single fixed path.
"""

import base64
import copy
import hashlib
import io
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:  # allows importing the module outside of ComfyUI
    folder_paths = None

# `from . import ...` only fails when there is no parent package to resolve --
# exactly what `__package__` already tells us -- so branching on it (rather
# than catching the ImportError) means a genuine failure inside
# atlascloud_models itself (e.g. a missing dependency) propagates with its
# real message instead of being masked by a second, misleading "no module
# named atlascloud_models" from a doomed flat-import retry.
if __package__:
    from . import atlascloud_models as _atlascloud_models
else:  # imported as a loose module (tests, tooling), not as a package submodule
    import atlascloud_models as _atlascloud_models

INPUT_KINDS = _atlascloud_models.INPUT_KINDS
OUTPUT_KINDS = _atlascloud_models.OUTPUT_KINDS
assign_media = _atlascloud_models.assign_media
build_payload = _atlascloud_models.build_payload
schema_for_model = _atlascloud_models.schema_for_model
image_kind_fields = _atlascloud_models.image_kind_fields
image_count_field = _atlascloud_models.image_count_field
provider_config = _atlascloud_models.provider_config
provider_label = _atlascloud_models.provider_label
submit_path = _atlascloud_models.submit_path
PROVIDER_IDS = _atlascloud_models.PROVIDER_IDS
DEFAULT_PROVIDER = _atlascloud_models.DEFAULT_PROVIDER


API_BASE = "https://api.atlascloud.ai/api/v1/model"
IMAGE_ENDPOINT = f"{API_BASE}/generateImage"
VIDEO_ENDPOINT = f"{API_BASE}/generateVideo"
UPLOAD_ENDPOINT = f"{API_BASE}/uploadMedia"
PREDICTION_ENDPOINT = f"{API_BASE}/prediction"

WAVESPEED_SERVER = "https://api.wavespeed.ai"
WAVESPEED_API_BASE = f"{WAVESPEED_SERVER}/api/v3"
WAVESPEED_UPLOAD_ENDPOINT = f"{WAVESPEED_API_BASE}/media/upload/binary"
WAVESPEED_PREDICTION_ENDPOINT = f"{WAVESPEED_API_BASE}/predictions"

# Per-provider endpoints. Atlas Cloud routes by output kind ("image" /
# "video") and carries the model in the body; WaveSpeed has a single submit
# URL per model, so its endpoint is built from the model id instead.
PROVIDER_ENDPOINTS = {
    "atlascloud": {
        "image": IMAGE_ENDPOINT,
        "video": VIDEO_ENDPOINT,
        "upload": UPLOAD_ENDPOINT,
        "prediction": PREDICTION_ENDPOINT,
        "prediction_suffix": "",
    },
    "wavespeed": {
        "image": "",
        "video": "",
        "upload": WAVESPEED_UPLOAD_ENDPOINT,
        "prediction": WAVESPEED_PREDICTION_ENDPOINT,
        "prediction_suffix": "/result",
    },
}

DONE_STATES = {"completed", "complete", "succeeded", "success", "finished"}
FAILED_STATES = {"failed", "error", "canceled", "cancelled", "timeout"}


def submit_endpoint(provider, model_identifier, output_kind, api_key=""):
    """The URL a generation request for this model has to be POSTed to.

    Atlas Cloud has one endpoint per output kind. WaveSpeed names the model in
    the path; the exact path comes from the catalogue where it is known, and
    falls back to the documented "/api/v3/<model id>" shape so an uncatalogued
    or brand-new model still works.
    """
    provider, _ = provider_config(provider)
    if provider != "wavespeed":
        return PROVIDER_ENDPOINTS[provider][output_kind]
    path = submit_path(model_identifier, provider=provider, api_key=api_key)
    if path:
        return f"{WAVESPEED_SERVER}{path}"
    return f"{WAVESPEED_API_BASE}/{model_identifier.strip().lstrip('/')}"


def build_headers(api_key, json_body=True, provider=DEFAULT_PROVIDER):
    label = provider_label(provider)
    if not api_key or not api_key.strip():
        raise ValueError(
            f"{label}: no API key provided. Paste the key from the {label} "
            f"console into the api_key widget."
        )
    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def parse_extra_params(extra_params, label="Atlas Cloud"):
    """Parse the free-form JSON widget into a dict of extra payload fields."""
    text = (extra_params or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{label}: extra_params is not valid JSON: {e}")
    if not isinstance(parsed, dict):
        raise ValueError(f"{label}: extra_params must be a JSON object, e.g. {{\"cfg\": 3.5}}")
    return parsed


def request_json(method, url, headers, timeout, label="Atlas Cloud", **kwargs):
    response = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(
            f"{label}: {method} {url} failed with HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    try:
        return response.json()
    except ValueError:
        raise RuntimeError(
            f"{label}: {method} {url} returned no JSON: {response.text[:300]}"
        )


def unwrap(payload):
    """Return the meaningful body: responses are wrapped in "data" or not."""
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload if isinstance(payload, dict) else {}


def extract_prediction_id(payload):
    body = unwrap(payload)
    for key in ("id", "prediction_id", "predictionId", "request_id", "requestId", "task_id"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    raise RuntimeError(f"Atlas Cloud: no prediction id in response: {json.dumps(payload)[:400]}")


def collect_urls(value, collected):
    """Walk the polymorphic output field and collect every media URL found."""
    if isinstance(value, str):
        if value.startswith(("http://", "https://", "data:")):
            collected.append(value)
    elif isinstance(value, list):
        for item in value:
            collect_urls(item, collected)
    elif isinstance(value, dict):
        for key in ("url", "image_url", "video_url", "signed_url", "output_url", "download_url"):
            if isinstance(value.get(key), str):
                collect_urls(value[key], collected)
                return
        for item in value.values():
            collect_urls(item, collected)


def extract_outputs(body):
    collected = []
    for key in ("outputs", "output", "images", "videos", "url", "image_url", "video_url", "download_url"):
        if key in body:
            collect_urls(body[key], collected)
        if collected:
            break
    # Deduplicate while keeping the original order.
    seen = set()
    return [url for url in collected if not (url in seen or seen.add(url))]


def upload_media(api_key, data_bytes, filename, mime_type, timeout, provider=DEFAULT_PROVIDER):
    provider, _ = provider_config(provider)
    label = provider_label(provider)
    headers = build_headers(api_key, json_body=False, provider=provider)
    payload = request_json(
        "POST", PROVIDER_ENDPOINTS[provider]["upload"], headers, timeout, label=label,
        files={"file": (filename, io.BytesIO(data_bytes), mime_type)},
    )
    body = unwrap(payload)
    # WaveSpeed answers with data.download_url, Atlas Cloud with url.
    for key in ("url", "file_url", "media_url", "signed_url", "download_url"):
        if isinstance(body.get(key), str) and body[key]:
            return body[key]
    raise RuntimeError(f"{label}: media upload returned no url: {json.dumps(payload)[:400]}")


def tensor_to_png_bytes(image_tensor):
    array = image_tensor.detach().cpu().numpy() if hasattr(image_tensor, "detach") else np.asarray(image_tensor)
    array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# Uploaded reference images are cached by content hash for the lifetime of the
# ComfyUI process: {sha256: (url, uploaded_at)}. Re-queueing a workflow with
# unchanged reference images therefore costs no further uploads. The entries
# expire after cache_minutes because Atlas Cloud does not document how long a
# media URL stays valid.
_upload_cache = {}


def upload_image_bytes_cached(api_key, png_bytes, timeout, cache_minutes, label,
                              provider=DEFAULT_PROVIDER):
    provider, _ = provider_config(provider)
    digest = hashlib.sha256(png_bytes).hexdigest()
    # Keyed by provider as well: an image uploaded to Atlas Cloud lives on a
    # host WaveSpeed cannot read, so the two must never share a cache entry.
    cache_key = (provider, digest)
    ttl = float(cache_minutes) * 60.0
    if ttl > 0:
        cached = _upload_cache.get(cache_key)
        if cached and (time.time() - cached[1]) < ttl:
            print(f"{label}: reusing uploaded reference {digest[:8]} ({cached[0]})")
            return cached[0]
    url = upload_media(
        api_key, png_bytes, f"comfyui_{uuid.uuid4().hex}.png", "image/png", timeout,
        provider=provider,
    )
    _upload_cache[cache_key] = (url, time.time())
    print(f"{label}: uploaded reference {digest[:8]} -> {url}")
    return url


def image_to_data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def collect_reference_images(api_key, image_batches, extra_urls_text, transport,
                             timeout, cache_minutes, label, provider=DEFAULT_PROVIDER):
    """Turn every connected IMAGE input into the reference list sent to the API.

    Each IMAGE input may itself be a batch (batches only hold images of equal
    size, which is why several separate inputs exist for differently sized
    references). Order is preserved: batch inputs first, then the manually
    supplied URLs.

    Returns (references, rejected_lines): a reference_urls line that is not a
    URL at all is dropped rather than sent, and named to the caller so the
    node can surface it. Sending it produced an error message that pointed
    nowhere near the cause -- a stray "wavespeed" reached Seedream as a
    reference and came back as "Invalid base64-encoded string: number of data
    characters (9) cannot be 1 more than a multiple of 4".
    """
    references = []
    for image_batch in image_batches:
        if image_batch is None:
            continue
        for index in range(image_batch.shape[0]):
            png_bytes = tensor_to_png_bytes(image_batch[index])
            if transport == "base64":
                references.append(image_to_data_uri(png_bytes))
            else:
                references.append(upload_image_bytes_cached(
                    api_key, png_bytes, timeout, cache_minutes, label, provider=provider
                ))
    rejected_lines = []
    for raw_line in (extra_urls_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # A comma separates several URLs on one line -- except inside a data:
        # URI, where the comma is what separates the media type from the
        # payload. Splitting those would cut every inlined image in half.
        candidates = [line] if line.startswith("data:") else line.split(",")
        for candidate in candidates:
            url = candidate.strip()
            if not url:
                continue
            if url.startswith(("http://", "https://", "data:")):
                references.append(url)
            else:
                rejected_lines.append(url)
    if rejected_lines:
        print(
            f"{label}: reference_urls holds {len(rejected_lines)} line(s) that are not URLs "
            f"({', '.join(repr(line[:40]) for line in rejected_lines)}); not sending them. "
            f"Expected http://, https:// or a data: URI, one per line."
        )
    return references, rejected_lines


def apply_reference_images(payload, references, image_field, label):
    """Write the references into the payload under the model's own field name.

    Field names differ per model family: `images` (Seedream edit, array of up
    to 10 URLs or base64 strings), `image` (step1x-edit, single URL),
    `image_url`/`image_urls` (video models). A plural field name always gets a
    list; a singular one gets a single string and warns when references are
    dropped.
    """
    if not references:
        return
    field = (image_field or "").strip() or "images"
    if field.endswith("s"):
        payload[field] = references
    else:
        payload[field] = references[0]
        if len(references) > 1:
            print(
                f"{label}: field '{field}' is singular, sending only the first of "
                f"{len(references)} reference images"
            )


def warn_if_input_map_may_misroute(image_batches, references, input_map, schema, model_identifier, label):
    """Flag the two input_map failure modes the browser cannot see coming.

    web/atlasCloudModels.js writes input_map indices that count *connected
    image slots* (see computeSingularInputMap there), while
    collect_reference_images above counts *images*: a slot fed by a
    multi-image batch contributes one reference per image, not one per slot.
    Whenever an earlier slot carries such a batch, every index after it is
    off by the size of that batch, so a mapping meant for "the last frame"
    can silently land on the wrong image. The browser has no way to inspect
    tensor shapes to catch this itself, so the check lives here, shared by
    both nodes, right before the mapping is applied.

    Only the *slot-derived* references (from the connected IMAGE inputs) are
    compared against connected_slots for this check -- references appended
    from reference_urls are not tied to any slot and would otherwise trip
    this warning too, blaming a batch that was never involved (see the
    second check below for what actually happens to those).
    """
    if not input_map:
        return
    connected_slots = sum(1 for batch in image_batches if batch is not None)
    slot_derived_reference_count = sum(
        batch.shape[0] for batch in image_batches if batch is not None
    )
    if slot_derived_reference_count > connected_slots:
        print(
            f"{label}: {model_identifier} input_map may misroute references -- "
            f"a connected image slot carries a multi-image batch, so slot "
            f"positions no longer line up with reference indices "
            f"({slot_derived_reference_count} references from {connected_slots} connected slot(s))"
        )

    # assign_media's input_map branch only ever writes the indices input_map
    # names; any collected reference at an index nothing points to (an extra
    # reference_urls entry, or a batch image beyond what the map covers) is
    # silently left out of the request. A user who pastes a URL into
    # reference_urls and never sees it reach the API needs to be told why,
    # not left guessing -- this does not change assign_media's own behaviour,
    # only reports on it.
    media_field_names = {field["name"] for field in image_kind_fields(schema)}
    mapped_indices = set()
    for field_name, index in input_map.items():
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(references) and field_name in media_field_names:
            mapped_indices.add(index)
    unused_count = len(references) - len(mapped_indices)
    if unused_count > 0:
        print(
            f"{label}: {model_identifier} input_map covers {len(mapped_indices)} of "
            f"{len(references)} collected reference(s); {unused_count} will not reach "
            f"the request -- check reference_urls and how many images each connected "
            f"image slot carries"
        )


def assign_reference_images(payload, schema, references, image_batches, image_field,
                             input_map, model_identifier, label):
    """Route collected reference images into the payload and report how.

    Shared by both nodes: `image_field` set explicitly overrides the schema
    (for models the schema lookup does not cover); left empty, references are
    routed through the schema's own image-kind media fields via assign_media
    (a video/audio field is never a target -- see image_kind_fields), with
    warn_if_input_map_may_misroute flagging the two ways input_map can go
    wrong first. Returns (media_fields, references_dropped) for the caller's
    `info` string -- `media_fields` names where the references went (or the
    override field, or "(none)"), `references_dropped` is True when the
    schema declares no image-kind media field at all despite references
    being connected (whether because it declares none whatsoever, or only
    video/audio ones this node cannot fill).
    """
    if image_field.strip():
        apply_reference_images(payload, references, image_field, label)
        return image_field, False

    references_dropped = False
    usable_fields = image_kind_fields(schema)
    if references and not usable_fields:
        references_dropped = True
        all_media_fields = schema.get("image_fields") or []
        if all_media_fields:
            # The schema declares media fields, just none that accept an
            # image -- a video model whose only media field is `video` or
            # `audio` (see the final review, Finding 4). Different message
            # from the "no media field at all" case below: this model *can*
            # take a reference, just not the kind that is connected.
            non_image_names = ", ".join(field["name"] for field in all_media_fields)
            print(
                f"{label}: {model_identifier!r} only declares non-image media field(s) "
                f"({non_image_names}); the {len(references)} connected reference image(s) "
                f"cannot be used by this model"
            )
        else:
            # The schema is real (not the fallback, which always declares a
            # media field) but simply has no media field at all -- without
            # this, assign_media() would no-op and the earlier "N reference
            # image(s) via upload" print would misleadingly suggest they
            # were sent somewhere.
            print(
                f"{label}: {model_identifier!r} declares no image fields; "
                f"the {len(references)} connected reference image(s) are not being sent"
            )
    warn_if_input_map_may_misroute(
        image_batches, references, input_map, schema, model_identifier, label
    )
    assign_media(payload, schema, references, input_map,
                 model_identifier=model_identifier, label=label)
    media_fields = ", ".join(field["name"] for field in usable_fields) or "(none)"
    return media_fields, references_dropped


def resolve_model_schema(model_identifier, label, fallback_prompt_field="prompt",
                          fallback_image_fields=None, fallback_fields=None,
                          provider=DEFAULT_PROVIDER, api_key=""):
    """Look up a model's request schema at the selected provider, tolerating
    the model being unknown to the catalogue or the catalogue being
    unreachable.

    `model` has always been free text (see the class docstring below), so a
    model Atlas has not indexed yet, a stale disk cache, or a momentary
    network outage must not crash the node. In that situation the prompt,
    reference images, and the node's own conventional fields (`prompt`,
    `images`, `width`/`height` for the image node; `prompt`, `image_url`,
    `duration`/`aspect_ratio`/`resolution` for the video node -- the shape
    each caller passes as `fallback_*`) -- the same raw fields the node
    always used before this feature existed -- are still routed instead of
    being silently dropped. When we genuinely know nothing about a model,
    passing the user's own values through and letting Atlas decide is
    strictly better than guessing wrong or dropping them; that is different
    from the "guessing" this feature eliminates, which was about overriding
    a schema we DO have. build_payload's own "this model takes no text"
    behaviour still applies normally whenever a real schema is found; the
    fallback below only kicks in for the sentinel "nothing known at all"
    shape.

    Returns (schema, used_fallback) so the caller can surface the fallback
    to the user instead of it only showing up as a console print.
    """
    try:
        schema = schema_for_model(model_identifier, provider=provider, api_key=api_key)
    except Exception as error:
        print(
            f"{label}: schema lookup for {model_identifier!r} failed ({error}); "
            f"falling back to this node's default fields"
        )
        schema = None
    schema_is_empty = not schema or not (
        schema.get("prompt_field") or schema.get("fields") or schema.get("image_fields")
    )
    if schema_is_empty:
        fallback_schema = {
            "model": model_identifier,
            "prompt_field": fallback_prompt_field,
            "image_fields": fallback_image_fields if fallback_image_fields is not None else [
                {"name": "images", "is_list": True, "max_items": 10}
            ],
            "fields": fallback_fields if fallback_fields is not None else [
                {"name": "width", "type": "integer", "default": None, "enum": None,
                 "required": False, "description": "", "is_media": False, "is_list": False},
                {"name": "height", "type": "integer", "default": None, "enum": None,
                 "required": False, "description": "", "is_media": False, "is_list": False},
            ],
        }
        return fallback_schema, True
    return schema, False


def download_bytes(url, timeout, label="Atlas Cloud"):
    if url.startswith("data:"):
        header, _, encoded = url.partition(",")
        if ";base64" in header:
            return base64.b64decode(encoded)
        return encoded.encode("utf-8")
    response = requests.get(url, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"{label}: download of {url[:120]} failed with HTTP {response.status_code}")
    return response.content


def poll_url(submit_response, prediction_id, provider=DEFAULT_PROVIDER):
    """Where to poll for this prediction.

    WaveSpeed hands back the exact result URL in data.urls.get; using it
    keeps working if the API version in that path ever moves. Both providers
    fall back to their documented "<prediction endpoint>/<id>" shape.
    """
    provider, _ = provider_config(provider)
    urls = unwrap(submit_response).get("urls")
    if isinstance(urls, dict) and isinstance(urls.get("get"), str) and urls["get"]:
        return urls["get"]
    endpoints = PROVIDER_ENDPOINTS[provider]
    return f"{endpoints['prediction']}/{prediction_id}{endpoints['prediction_suffix']}"


def submit_and_poll(endpoint, api_key, payload, poll_interval, timeout_seconds, request_timeout,
                    label, provider=DEFAULT_PROVIDER, model_identifier=""):
    """Submit a generation request and poll until the prediction is terminal."""
    provider, _ = provider_config(provider)
    error_label = provider_label(provider)
    headers = build_headers(api_key, provider=provider)
    # WaveSpeed names the model in the URL, so the body has no "model" key to
    # report -- the caller passes the id in that case.
    print(f"{label}: submitting model={(payload.get('model') or model_identifier)!r}")
    submit_response = request_json(
        "POST", endpoint, headers, request_timeout, label=error_label, json=payload
    )
    prediction_id = extract_prediction_id(submit_response)
    print(f"{label}: prediction {prediction_id}, polling every {poll_interval:.1f}s")

    status_url = poll_url(submit_response, prediction_id, provider)
    started = time.time()
    last_status = ""
    while True:
        elapsed = time.time() - started
        if elapsed > timeout_seconds:
            raise RuntimeError(
                f"{label}: prediction {prediction_id} still '{last_status or 'unknown'}' "
                f"after {elapsed:.0f}s (timeout {timeout_seconds}s)"
            )
        body = unwrap(request_json("GET", status_url, headers, request_timeout, label=error_label))
        status = str(body.get("status", "")).lower()
        if status != last_status:
            print(f"{label}: status={status or 'unknown'} ({elapsed:.0f}s)")
            last_status = status
        if status in FAILED_STATES:
            raise RuntimeError(
                f"{label}: prediction {prediction_id} failed: "
                f"{body.get('error') or json.dumps(body)[:300]}"
            )
        if status in DONE_STATES:
            outputs = extract_outputs(body)
            if not outputs:
                raise RuntimeError(
                    f"{label}: prediction {prediction_id} completed without outputs: "
                    f"{json.dumps(body)[:400]}"
                )
            print(f"{label}: done after {elapsed:.0f}s, {len(outputs)} output(s)")
            return outputs, prediction_id, elapsed
        time.sleep(poll_interval)


def images_to_tensor(url_list, timeout, label):
    """Download images and stack them into a single ComfyUI IMAGE batch."""
    return image_bytes_to_tensor([download_bytes(url, timeout) for url in url_list], label)


def image_bytes_to_tensor(blob_list, label):
    """Stack already downloaded image bytes into a single ComfyUI IMAGE batch.

    Split out of images_to_tensor so a parallel run can download each
    prediction's outputs inside its own worker thread (see
    AtlasCloudImage.generate) and still stack everything into one batch here,
    in prompt order, on the calling thread.
    """
    frames = []
    target_size = None
    for blob in blob_list:
        image = Image.open(io.BytesIO(blob)).convert("RGB")
        if target_size is None:
            target_size = image.size
        elif image.size != target_size:
            print(f"{label}: resizing {image.size} output to {target_size} to keep one batch")
            image = image.resize(target_size, Image.LANCZOS)
        frames.append(np.asarray(image).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(frames))


def first_value(value, default=None):
    """Collapse an INPUT_IS_LIST input back to the single value it carries.

    AtlasCloudImage declares INPUT_IS_LIST so it receives a whole prompt list
    in one execution (see the class docstring). ComfyUI then hands over *every*
    input as a list, widgets included - each of those carries exactly one
    entry. Called with a plain value (tests, direct calls) it passes through
    unchanged, so the node works both ways.
    """
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def prompt_values(value):
    """The prompts one execution has to render, in order.

    A single prompt stays exactly as it is - including an empty one, which is
    legitimate for models that declare no prompt field. Blank entries are only
    dropped from an actual list, where they are the trailing empty lines of a
    prompt-list node rather than a deliberate "send no prompt".
    """
    values = value if isinstance(value, list) else [value]
    prompts = [str(item) for item in values if item is not None]
    if len(prompts) > 1:
        return [prompt for prompt in prompts if prompt.strip()] or [""]
    return prompts or [""]


def image_batches_of(values):
    """Flatten the IMAGE inputs into the batch list the reference helpers take.

    Under INPUT_IS_LIST each connected slot arrives as a list of tensors
    (normally one). Every tensor is kept, in slot order, so a slot fed by a
    list-producing node contributes all of its images as references.
    """
    batches = []
    for value in values:
        for batch in (value if isinstance(value, list) else [value]):
            if batch is not None:
                batches.append(batch)
    return batches


def video_bytes_to_frames(video_bytes, max_frames, frame_stride, label):
    """Decode video bytes into an IMAGE batch. Tries PyAV, then OpenCV."""
    temp_dir = folder_paths.get_temp_directory() if folder_paths else "."
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, f"atlascloud_{uuid.uuid4().hex}.mp4")
    with open(video_path, "wb") as handle:
        handle.write(video_bytes)

    stride = max(1, int(frame_stride))
    limit = int(max_frames) if int(max_frames) > 0 else None
    frames = []
    fps = 0.0

    try:
        import av

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            fps = float(rate) if rate else 0.0
            for index, frame in enumerate(container.decode(video=0)):
                if index % stride:
                    continue
                frames.append(frame.to_ndarray(format="rgb24"))
                if limit and len(frames) >= limit:
                    break
    except ImportError:
        import cv2

        capture = cv2.VideoCapture(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if limit and len(frames) >= limit:
                        break
                index += 1
        finally:
            capture.release()

    if not frames:
        raise RuntimeError(f"{label}: video decoded to zero frames ({video_path})")

    print(f"{label}: decoded {len(frames)} frames at {fps:.2f} fps (stride {stride})")
    batch = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(batch), fps / stride, video_path


class AtlasCloudImage:
    """
    Generates or edits an image through the Atlas Cloud or WaveSpeed API and
    returns it as a ComfyUI IMAGE batch.

    `provider` selects which of the two hosted APIs the request goes to; the
    api_key, the model list and the model id all belong to that provider. It
    is declared as the very last widget so a workflow saved before it existed
    still loads with every other value in place, on the Atlas Cloud default.

    All request fields are node inputs. `model` is free text so any model id of
    the selected provider works without updating this node (e.g.
    "bytedance/seedream-v5.0-pro/edit" on Atlas Cloud, or
    "wavespeed-ai/flux-2-pro/text-to-image" on WaveSpeed).

    Numeric fields are omitted from the request when set to 0 (seed: -1), so a
    model's own defaults stay in effect unless a value is chosen explicitly.
    `width`/`height` are converted into whatever the model's own schema
    declares (e.g. a single `size` string such as "1328*1776" for Seedream,
    or separate `width`/`height` keys for models that take those). Anything
    the model supports beyond the built-in widgets goes into `params_json`
    (schema-driven, filled in by the web extension) and/or `extra_params`
    (free-form JSON); both are merged into the request, with `extra_params`
    applied last so it always wins.

    `output_type`/`input_type` only filter the model list the web extension
    offers and are never sent to the API.

    Reference images: the ten IMAGE inputs (`image`, `image_2` .. `image_10`)
    are concatenated in order, so up to ten differently sized references can
    be attached (a single IMAGE input is already a batch, but a batch can only
    hold images of equal size - hence the separate inputs). `reference_urls`
    appends already hosted images, one URL per line. `input_map` optionally
    pins a specific reference index to a schema field name (e.g. which image
    is the "last frame"); left empty, references fill the schema's media
    fields in order.

    The receiving field name is normally read from the model's own schema.
    `image_field` remains an escape hatch: when non-empty it overrides the
    schema and writes references there instead, for models the schema lookup
    does not (yet) cover. A plural name receives the list, a singular one the
    first reference only.

    `image_transport` selects how the references travel: `upload` pushes them
    through uploadMedia and sends the resulting links (payload stays small),
    `base64` inlines them as data URIs without any upload - supported wherever
    the docs say a field takes "a URL or base64 encoded image".

    Uploads are cached process-wide by image content hash for
    `upload_cache_minutes`, so re-running a workflow with unchanged references
    re-uploads nothing. Set 0 to disable the cache.

    Prompt lists and `max_parallel`: the node declares INPUT_IS_LIST, so a
    prompt list (e.g. from "CR Prompt List") arrives complete in a single
    execution instead of making ComfyUI run the node once per prompt,
    strictly one after another. `max_parallel` is how many of those prompts
    are in flight at Atlas Cloud at the same time; 1 keeps the old sequential
    behaviour. Reference images are collected and uploaded once and shared by
    every request. The node returns when all requests are done, with every
    generated image in one IMAGE batch in prompt order.

    A request that fails does not take the batch down: the images that did
    come back are returned and `Info` names the failed prompts. Only when
    every request fails does the node raise.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # provider comes first because it decides what every widget
                # below it means -- which key is valid, which catalogue the
                # model dropdown shows. It used to be declared last, so that
                # workflows saved before it existed kept loading; that is now
                # handled by the widgets_values migration in
                # web/atlasCloudModels.js instead (widgets_values is a
                # positional array, so moving a widget shifts every value
                # after it).
                "provider": (PROVIDER_IDS, {
                    "default": DEFAULT_PROVIDER,
                    "tooltip": "Which hosted API to send this request to. The "
                               "api_key and the model list belong to the "
                               "selected provider.",
                }),
                "api_key": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "API key of the selected provider (its console -> API keys).",
                }),
                "model": ("STRING", {
                    "multiline": False, "default": "bytedance/seedream-v5.0-pro/text-to-image",
                    "tooltip": "Model id of the selected provider, taken from the "
                               "model page URL.",
                }),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "0 = omit, model default applies.",
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "0 = omit, model default applies.",
                }),
                "num_images": ("INT", {
                    "default": 1, "min": 1, "max": 16,
                    "tooltip": "Sent under whichever name the model declares "
                               "(num_images / n / max_images). Most image models "
                               "declare none and always return one image - the "
                               "Info output says so when that happens.",
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xFFFFFFFF,
                    "tooltip": "-1 = no seed sent and the node re-runs on every "
                               "queue; any other value is sent to the API.",
                }),
                "poll_interval": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 30.0, "step": 0.5}),
                "timeout": ("INT", {
                    "default": 300, "min": 10, "max": 3600,
                    "tooltip": "Seconds to wait for the prediction before failing.",
                }),
                "image_field": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "Leave empty to use the field names the selected "
                               "model's schema declares (recommended). Set a name "
                               "to override: images (Seedream edit), image "
                               "(step1x-edit), image_url / image_urls (others) - a "
                               "plural name gets the list, a singular one the "
                               "first image.",
                }),
                "image_transport": (["upload", "base64"], {
                    "default": "upload",
                    "tooltip": "upload = send via uploadMedia and pass the URLs; "
                               "base64 = inline the images as data URIs.",
                }),
                "upload_cache_minutes": ("INT", {
                    "default": 120, "min": 0, "max": 10080,
                    "tooltip": "Reuse an uploaded reference with identical content "
                               "for this many minutes. 0 = upload every run.",
                }),
                "extra_params": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "JSON object merged into the request body, e.g. "
                               "{\"size\": \"2048*2048\", \"max_images\": 2}",
                }),
                "params_json": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": "Model parameters taken from the model's Atlas schema, filled "
                               "in by the widgets above. Merged before extra_params.",
                }),
                "input_map": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "Optional JSON mapping a schema media field to an image slot "
                               "index, e.g. {\"last_image\": 1}. Empty = schema order.",
                }),
                # Placed last (not first) so a workflow saved by the pre-schema
                # node -- which serializes widgets_values positionally and knows
                # nothing about these two -- still loads every other value into
                # the widget it belongs to; see the widget-order note on
                # AtlasCloudVideo below for the full rationale.
                "output_type": (OUTPUT_KINDS, {
                    "default": "Image",
                    "tooltip": "Filters the model list; not sent to the API.",
                }),
                "input_type": (["Any"] + INPUT_KINDS, {
                    "default": "Any",
                    "tooltip": "Filters the model list by what the model accepts as input.",
                }),
                # Appended after output_type/input_type for the same reason
                # they were moved to the end: widgets_values is positional, so
                # a new required widget may only ever be added last.
                "max_parallel": ("INT", {
                    "default": 1, "min": 1, "max": 32,
                    "tooltip": "How many prompts of a connected prompt list are "
                               "sent to the provider at the same time. 1 = one "
                               "after another. Has no effect on a single prompt.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Reference image(s) for edit / image-to-image models. "
                               "A batch counts as several references.",
                }),
                "image_2": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_3": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_4": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_5": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_6": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_7": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_8": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_9": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_10": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "reference_urls": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Already hosted reference images, one URL per line; "
                               "appended after the connected images.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("Images", "URLs", "Info",)
    FUNCTION = "generate"
    CATEGORY = "AtlasCloud"
    # A prompt list must reach generate() whole, otherwise ComfyUI executes the
    # node once per prompt and nothing can ever run in parallel. Every input
    # therefore arrives as a list and is collapsed with first_value below.
    INPUT_IS_LIST = True

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        # Without an explicit seed the API result is not reproducible, so the
        # node must re-run instead of serving the cached image.
        seed_value = int(first_value(seed, -1))
        return float("nan") if seed_value < 0 else seed_value

    def generate(self, api_key, output_type, input_type, model, prompt, width, height,
                 num_images, seed, poll_interval, timeout, image_field, image_transport,
                 upload_cache_minutes, extra_params, params_json="{}", input_map="",
                 max_parallel=1, provider=DEFAULT_PROVIDER,
                 image=None, image_2=None, image_3=None, image_4=None, image_5=None,
                 image_6=None, image_7=None, image_8=None, image_9=None, image_10=None,
                 reference_urls=""):
        # output_type/input_type only filter the model list the web extension
        # offers; they are never part of the request body.
        provider, _ = provider_config(first_value(provider, DEFAULT_PROVIDER))
        label = "AtlasCloudImage" if provider == "atlascloud" else "WaveSpeedImage"
        # INPUT_IS_LIST: `prompt` is the actual list to work through, every
        # other input is a one-entry list carrying its widget value.
        prompts = prompt_values(prompt)
        api_key = first_value(api_key, "")
        model = first_value(model, "")
        width = first_value(width, 0)
        height = first_value(height, 0)
        num_images = first_value(num_images, 1)
        seed = first_value(seed, -1)
        poll_interval = first_value(poll_interval, 2.0)
        timeout = first_value(timeout, 300)
        image_field = first_value(image_field, "") or ""
        image_transport = first_value(image_transport, "upload")
        upload_cache_minutes = first_value(upload_cache_minutes, 120)
        extra_params = first_value(extra_params, "")
        params_json = first_value(params_json, "{}")
        input_map = first_value(input_map, "")
        reference_urls = first_value(reference_urls, "")
        parallel_limit = max(1, int(first_value(max_parallel, 1)))
        model_identifier = model.strip()
        schema, schema_is_fallback = resolve_model_schema(
            model_identifier, label, provider=provider, api_key=api_key
        )

        parameters = parse_extra_params(params_json, label)
        # The node's own seed/num_images widgets win over whatever
        # params_json (the schema-driven param_seed/param_num_images
        # widgets) carries when the user actually set them -- not just
        # setdefault, an explicit overwrite with a diagnostic when it changes
        # something. These two are the widgets ComfyUI's own caching
        # (IS_CHANGED below) and "control_after_generate" are tied to, so a
        # schema value quietly winning here used to make ComfyUI cache a
        # result as seed-reproducible while a different seed was actually
        # sent (see the final review, Finding 3).
        #
        # The widget is called num_images, but the field name a model takes
        # for it varies (num_images / n / max_images) and most image models
        # declare none at all -- see image_count_field. Sending "num_images"
        # regardless made the request carry a key the API ignores: a
        # num_images=2 run on bytedance/seedream-v5.0-pro/edit returned a
        # single image with nothing but a console line to explain it. The
        # value is therefore routed to the name the selected model declares,
        # and a model that can only ever return one image says so in `info`,
        # which the workflow actually shows.
        count_warning = ""
        if int(num_images) > 1:
            # An uncatalogued model (fallback schema) declares nothing at all,
            # so the pre-schema behaviour is kept there: send "num_images" and
            # let the API decide, rather than claim the model cannot do it.
            count_field = "num_images" if schema_is_fallback else image_count_field(schema)
            if count_field is None:
                count_warning = (
                    f"num_images={int(num_images)} not sent: {model_identifier} declares "
                    f"no image count field and returns a single image"
                )
                print(f"{label}: {count_warning}")
            else:
                if count_field in parameters and parameters[count_field] != int(num_images):
                    print(
                        f"{label}: params_json {count_field}={parameters[count_field]!r} overridden "
                        f"by the node's own num_images widget ({int(num_images)})"
                    )
                if count_field != "num_images":
                    print(
                        f"{label}: num_images={int(num_images)} sent as {count_field!r}, "
                        f"the field {model_identifier} declares for it"
                    )
                parameters[count_field] = int(num_images)
        if int(seed) >= 0:
            if "seed" in parameters and parameters["seed"] != int(seed):
                print(
                    f"{label}: params_json seed={parameters['seed']!r} overridden by the "
                    f"node's own seed widget ({int(seed)})"
                )
            parameters["seed"] = int(seed)

        # The request body is assembled once, from the first prompt, so every
        # schema/reference diagnostic below is printed once instead of once
        # per prompt; the remaining prompts only swap the prompt field of a
        # copy (see request_for_prompt). Everything else - parameters,
        # references, extra_params - is identical across the batch by
        # construction.
        base_payload = build_payload(
            model_identifier, prompts[0], parameters,
            width=int(width), height=int(height), schema=schema, provider=provider,
        )

        image_batches = image_batches_of([image, image_2, image_3, image_4, image_5,
                                          image_6, image_7, image_8, image_9, image_10])
        references, rejected_reference_lines = collect_reference_images(
            api_key, image_batches, reference_urls, image_transport,
            int(timeout), int(upload_cache_minutes), label, provider=provider,
        )
        if references:
            print(f"{label}: {len(references)} reference image(s) via {image_transport}")

        input_map_dict = parse_extra_params(input_map, label) or None
        media_fields, references_dropped = assign_reference_images(
            base_payload, schema, references, image_batches, image_field,
            input_map_dict, model_identifier, label,
        )

        base_payload.update(parse_extra_params(extra_params, label))

        endpoint = submit_endpoint(provider, model_identifier, "image", api_key)
        prompt_field = schema.get("prompt_field") or ""
        total = len(prompts)

        def request_for_prompt(index):
            if index == 0:
                return base_payload
            payload = copy.deepcopy(base_payload)
            if prompt_field:
                payload[prompt_field] = prompts[index]
            return payload

        def run_request(index):
            # Each worker downloads its own outputs as well, so the downloads
            # overlap with the other predictions instead of queuing up behind
            # them once everything has been generated.
            request_label = label if total == 1 else f"{label}[{index + 1}/{total}]"
            urls, prediction_id, elapsed = submit_and_poll(
                endpoint, api_key, request_for_prompt(index), float(poll_interval),
                int(timeout), request_timeout=180, label=request_label,
                provider=provider, model_identifier=model_identifier,
            )
            return {
                "urls": urls,
                "blobs": [download_bytes(url, int(timeout), label) for url in urls],
                "prediction_id": prediction_id,
                "elapsed": elapsed,
            }

        workers = min(parallel_limit, total)
        if total > 1:
            print(f"{label}: {total} prompts, up to {workers} request(s) in parallel")
        started = time.time()
        results_by_index = {}
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_request, index): index for index in range(total)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results_by_index[index] = future.result()
                except Exception as error:
                    # Best effort: one failed prompt must not throw away the
                    # images the other requests already paid for. The whole
                    # batch only fails when nothing at all came back.
                    failures.append((index, f"{type(error).__name__}: {error}", error))
                    print(f"{label}: prompt {index + 1}/{total} failed: {error}")
        wall_clock = time.time() - started

        if not results_by_index:
            if total == 1:
                # A single prompt is the pre-parallel case: raise the API's own
                # error unchanged rather than a summary wrapped around one item.
                raise failures[0][2]
            raise RuntimeError(
                f"{label}: all {total} requests failed:\n"
                + "\n".join(f"  [{index + 1}] {message}" for index, message, _ in sorted(failures))
            )

        ordered = [results_by_index[index] for index in sorted(results_by_index)]
        urls = [url for result in ordered for url in result["urls"]]
        images = image_bytes_to_tensor(
            [blob for result in ordered for blob in result["blobs"]], label
        )
        predictions = ", ".join(result["prediction_id"] for result in ordered)
        info = (
            f"prediction={predictions}\nprovider={provider}\nmodel={model_identifier}\n"
            f"references={len(references)} field={media_fields}\n"
            f"images={images.shape[0]} size={images.shape[2]}x{images.shape[1]}\n"
            f"elapsed={wall_clock:.1f}s"
        )
        if total > 1:
            info += (
                f"\nprompts={len(ordered)}/{total} done, max_parallel={parallel_limit}"
            )
        if failures:
            info += "\nfailed prompts:\n" + "\n".join(
                f"  [{index + 1}] {message}" for index, message, _ in sorted(failures)
            )
        if count_warning:
            info += f"\n{count_warning}"
        if schema_is_fallback:
            info += "\nschema=unknown (fallback: prompt/images/width/height)"
        if references_dropped:
            info += f"\nreferences dropped: {model_identifier} declares no image fields"
        if rejected_reference_lines:
            info += (
                f"\nreference_urls: {len(rejected_reference_lines)} non-URL line(s) not sent: "
                + ", ".join(repr(line[:40]) for line in rejected_reference_lines)
            )
        return (images, "\n".join(urls), info)


# Fallback schema used when a video model is unknown to the Atlas catalogue
# (see resolve_model_schema): the same "prompt/duration/aspect_ratio/
# resolution/image_url" fields the pre-schema node always sent unconditionally,
# so a saved workflow on an uncatalogued model keeps working exactly as before.
VIDEO_FALLBACK_IMAGE_FIELDS = [{"name": "image_url", "is_list": False, "max_items": 1}]
VIDEO_FALLBACK_FIELDS = [
    {"name": "duration", "type": "integer", "default": None, "enum": None,
     "required": False, "description": "", "is_media": False, "is_list": False},
    {"name": "aspect_ratio", "type": "string", "default": None, "enum": None,
     "required": False, "description": "", "is_media": False, "is_list": False},
    {"name": "resolution", "type": "string", "default": None, "enum": None,
     "required": False, "description": "", "is_media": False, "is_list": False},
]


class AtlasCloudVideo:
    """
    Generates a video through the Atlas Cloud or WaveSpeed API and returns its
    frames as a ComfyUI IMAGE batch, so the result can be piped into any
    downstream video node (interpolation, upscaling, VHS combine, ...).

    `provider` selects which of the two hosted APIs the request goes to; the
    api_key, the model list and the model id all belong to that provider. It
    is declared as the very last widget so a workflow saved before it existed
    still loads with every other value in place, on the Atlas Cloud default.

    All request fields are node inputs. `model` is free text so any video model
    id of the selected provider works without updating this node (e.g.
    "kwaivgi/kling-v2.5-turbo-pro/image-to-video" or "google/veo-3.1").

    `duration`/`aspect_ratio`/`resolution` are sent only when the selected
    model's own schema declares that field (0 / empty always means "omit"
    regardless). For a model the provider has not indexed yet, a fallback schema
    keeps sending all three unconditionally -- the same behaviour this node
    had before schemas existed -- so saved workflows on such models are
    unaffected. Anything the model supports beyond the built-in widgets goes
    into `params_json` (schema-driven, filled in by the web extension)
    and/or `extra_params` (free-form JSON); both are merged into the request,
    with `extra_params` applied last so it always wins.

    `output_type`/`input_type` only filter the model list the web extension
    offers and are never sent to the API.

    Reference images: the ten IMAGE inputs (`image`, `image_2` .. `image_10`)
    are concatenated in order, so up to ten differently sized references can
    be attached (a single IMAGE input is already a batch, but a batch can only
    hold images of equal size - hence the separate inputs). This is where
    video models like Kling take a first and a last frame as two distinct
    fields (`image` / `last_image`) rather than one list. `reference_urls`
    appends already hosted images, one URL per line. `input_map` optionally
    pins a specific reference index to a schema field name (e.g. which image
    is the last frame); left empty, references fill the schema's media
    fields in order. Because `input_map`'s indices count connected image
    slots while a slot fed by a multi-image batch contributes several
    references, a console message flags the (rare) case where those two
    counts disagree instead of silently misrouting a frame.

    The receiving field name is normally read from the model's own schema.
    `image_field` remains an escape hatch: when non-empty it overrides the
    schema and writes references there instead, for models the schema lookup
    does not (yet) cover. A plural name receives the list, a singular one the
    first reference only.

    `image_transport` selects how the references travel: `upload` pushes them
    through uploadMedia and sends the resulting links (payload stays small),
    `base64` inlines them as data URIs without any upload - supported wherever
    the docs say a field takes "a URL or base64 encoded image".

    Uploads are cached process-wide by image content hash for
    `upload_cache_minutes`, so re-running a workflow with unchanged references
    re-uploads nothing. Set 0 to disable the cache.

    Decoding uses PyAV (shipped with ComfyUI) and falls back to OpenCV. The
    downloaded mp4 is kept in the ComfyUI temp directory and its path is
    returned so it can be saved or inspected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # First for the same reason as on the image node above; the
                # widgets_values shift this causes for older workflows is
                # migrated in web/atlasCloudModels.js.
                "provider": (PROVIDER_IDS, {
                    "default": DEFAULT_PROVIDER,
                    "tooltip": "Which hosted API to send this request to. The "
                               "api_key and the model list belong to the "
                               "selected provider.",
                }),
                "api_key": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "API key of the selected provider (its console -> API keys).",
                }),
                "model": ("STRING", {
                    "multiline": False, "default": "kwaivgi/kling-v2.5-turbo-pro/image-to-video",
                    "tooltip": "Model id of the selected provider, taken from the "
                               "model page URL.",
                }),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {
                    "default": 0, "min": 0, "max": 60,
                    "tooltip": "Seconds; 0 = omit, model default applies. Sent only "
                               "if the selected model's schema declares a duration field.",
                }),
                "aspect_ratio": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "e.g. 16:9; empty = omit. Sent only if the selected "
                               "model's schema declares an aspect_ratio field.",
                }),
                "resolution": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "e.g. 720p / 1080p; empty = omit. Sent only if the "
                               "selected model's schema declares a resolution field.",
                }),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 0xFFFFFFFF,
                    "tooltip": "-1 = no seed sent and the node re-runs on every "
                               "queue; any other value is sent to the API.",
                }),
                "poll_interval": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "timeout": ("INT", {
                    "default": 900, "min": 30, "max": 7200,
                    "tooltip": "Seconds to wait for the prediction before failing.",
                }),
                "max_frames": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "0 = decode all frames; otherwise stop after N frames "
                               "(guards VRAM/RAM on long clips).",
                }),
                "frame_stride": ("INT", {
                    "default": 1, "min": 1, "max": 16,
                    "tooltip": "Keep every Nth frame; the reported FPS is divided accordingly.",
                }),
                "image_field": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "Leave empty to use the field names the selected "
                               "model's schema declares (recommended). Set a name "
                               "to override: image_url (Kling and most video "
                               "models), image, or a plural name such as "
                               "image_urls for models that take several reference "
                               "frames - a plural name gets the list, a singular "
                               "one the first image.",
                }),
                "image_transport": (["upload", "base64"], {
                    "default": "upload",
                    "tooltip": "upload = send via uploadMedia and pass the URLs; "
                               "base64 = inline the images as data URIs.",
                }),
                "upload_cache_minutes": ("INT", {
                    "default": 120, "min": 0, "max": 10080,
                    "tooltip": "Reuse an uploaded image with identical content for "
                               "this many minutes. 0 = upload every run.",
                }),
                "extra_params": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "JSON object merged into the request body, e.g. "
                               "{\"negative_prompt\": \"blurry\", \"cfg_scale\": 0.5}",
                }),
                "params_json": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": "Model parameters taken from the model's Atlas schema, filled "
                               "in by the widgets above. Merged before extra_params.",
                }),
                "input_map": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "Optional JSON mapping a schema media field to an image slot "
                               "index, e.g. {\"last_image\": 1}. Empty = schema order.",
                }),
                # output_type/input_type are placed last, not first as they were
                # originally, because ComfyUI serializes a saved workflow's
                # widgets_values as a positional array and restores it by
                # declaration order on load. Putting them first shifted every
                # other value (model into output_type, prompt into input_type,
                # duration into model, ...) for any workflow saved by the
                # pre-schema node, which never had these two widgets at all.
                # Appending them after every widget that node already declared
                # keeps indices 0..(len(old required) - 1) identical to the old
                # node, so an old workflow's values land back in the same
                # widgets; only output_type/input_type themselves start unset
                # on such a load, which is harmless (they only filter the model
                # dropdown, never reach the request).
                "output_type": (OUTPUT_KINDS, {
                    "default": "Video",
                    "tooltip": "Filters the model list; not sent to the API.",
                }),
                "input_type": (["Any"] + INPUT_KINDS, {
                    "default": "Any",
                    "tooltip": "Filters the model list by what the model accepts as input.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "First frame for image-to-video models; uploaded "
                               "automatically. A batch counts as several references.",
                }),
                "image_2": ("IMAGE", {
                    "tooltip": "Second image, e.g. the last frame for models that "
                               "take a start and an end frame.",
                }),
                "image_3": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_4": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_5": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_6": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_7": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_8": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_9": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "image_10": ("IMAGE", {"tooltip": "Further reference(s), any size."}),
                "reference_urls": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Already hosted images, one URL per line; appended "
                               "after the connected images.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("Frames", "FPS", "VideoURL", "VideoPath", "Info",)
    FUNCTION = "generate"
    CATEGORY = "AtlasCloud"

    @classmethod
    def IS_CHANGED(cls, seed=-1, **kwargs):
        return float("nan") if int(seed) < 0 else seed

    def generate(self, api_key, output_type, input_type, model, prompt, duration,
                 aspect_ratio, resolution, seed, poll_interval, timeout, max_frames,
                 frame_stride, image_field, image_transport, upload_cache_minutes,
                 extra_params, params_json="{}", input_map="", provider=DEFAULT_PROVIDER,
                 image=None, image_2=None, image_3=None, image_4=None, image_5=None,
                 image_6=None, image_7=None, image_8=None, image_9=None, image_10=None,
                 reference_urls=""):
        # output_type/input_type only filter the model list the web extension
        # offers; they are never part of the request body.
        provider, _ = provider_config(provider)
        label = "AtlasCloudVideo" if provider == "atlascloud" else "WaveSpeedVideo"
        model_identifier = model.strip()
        schema, schema_is_fallback = resolve_model_schema(
            model_identifier, label,
            fallback_image_fields=VIDEO_FALLBACK_IMAGE_FIELDS,
            fallback_fields=VIDEO_FALLBACK_FIELDS,
            provider=provider, api_key=api_key,
        )

        parameters = parse_extra_params(params_json, label)
        # The legacy widgets still work, but only for models whose schema (or
        # the fallback above, for uncatalogued models) declares the field.
        declared = {field["name"] for field in schema.get("fields") or []}
        if int(duration) > 0 and "duration" in declared:
            parameters.setdefault("duration", int(duration))
        if aspect_ratio.strip() and "aspect_ratio" in declared:
            parameters.setdefault("aspect_ratio", aspect_ratio.strip())
        if resolution.strip() and "resolution" in declared:
            parameters.setdefault("resolution", resolution.strip())
        # The node's own seed widget wins over params_json's when the user
        # set it -- see the matching comment in AtlasCloudImage.generate.
        if int(seed) >= 0:
            if "seed" in parameters and parameters["seed"] != int(seed):
                print(
                    f"{label}: params_json seed={parameters['seed']!r} overridden by the "
                    f"node's own seed widget ({int(seed)})"
                )
            parameters["seed"] = int(seed)

        payload = build_payload(model_identifier, prompt, parameters, schema=schema,
                                provider=provider)

        image_batches = [image, image_2, image_3, image_4, image_5,
                         image_6, image_7, image_8, image_9, image_10]
        references, rejected_reference_lines = collect_reference_images(
            api_key, image_batches, reference_urls, image_transport,
            int(timeout), int(upload_cache_minutes), label, provider=provider,
        )
        if references:
            print(f"{label}: {len(references)} reference image(s) via {image_transport}")

        input_map_dict = parse_extra_params(input_map, label) or None
        media_fields, references_dropped = assign_reference_images(
            payload, schema, references, image_batches, image_field,
            input_map_dict, model_identifier, label,
        )

        payload.update(parse_extra_params(extra_params, label))

        urls, prediction_id, elapsed = submit_and_poll(
            submit_endpoint(provider, model_identifier, "video", api_key),
            api_key, payload, float(poll_interval), int(timeout),
            request_timeout=120, label=label,
            provider=provider, model_identifier=model_identifier,
        )
        video_url = urls[0]
        video_bytes = download_bytes(video_url, int(timeout), label)
        frames, fps, video_path = video_bytes_to_frames(
            video_bytes, max_frames, frame_stride, label
        )
        info = (
            f"prediction={prediction_id}\nprovider={provider}\nmodel={model_identifier}\n"
            f"references={len(references)} field={media_fields}\n"
            f"frames={frames.shape[0]} size={frames.shape[2]}x{frames.shape[1]} fps={fps:.2f}\n"
            f"elapsed={elapsed:.1f}s\nfile={video_path}"
        )
        if schema_is_fallback:
            info += "\nschema=unknown (fallback: prompt/duration/aspect_ratio/resolution/image_url)"
        if references_dropped:
            info += f"\nreferences dropped: {model_identifier} declares no image fields"
        if rejected_reference_lines:
            info += (
                f"\nreference_urls: {len(rejected_reference_lines)} non-URL line(s) not sent: "
                + ", ".join(repr(line[:40]) for line in rejected_reference_lines)
            )
        return (frames, float(fps), video_url, video_path, info)


NODE_CLASS_MAPPINGS = {
    "AtlasCloudImage": AtlasCloudImage,
    "AtlasCloudVideo": AtlasCloudVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AtlasCloudImage": "Atlas Cloud / WaveSpeed Image",
    "AtlasCloudVideo": "Atlas Cloud / WaveSpeed Video",
}
