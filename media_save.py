"""ComfyUI output nodes for provenance-aware image and video files.

The final media file is built in this order:

1. Encode the image/video.
2. Embed standalone XMP and optional ComfyUI prompt/workflow metadata.
3. Optionally sign that exact file with C2PA.

Signing is deliberately fail-closed. If signing was requested, an unsigned file
is never returned as if the operation had succeeded.
"""

from __future__ import annotations

import datetime as _datetime
import html
import importlib
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

try:
    import folder_paths
except ImportError:  # Allows unit tests and standalone tooling without ComfyUI.
    folder_paths = None


AI_SOURCE_TYPES = {
    "Fully AI-generated": (
        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    ),
    "AI-edited / composite": (
        "http://cv.iptc.org/newscodes/digitalsourcetype/"
        "compositeWithTrainedAlgorithmicMedia"
    ),
}

_MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "mp4": "video/mp4",
}

_COUNTER_LOCK = threading.Lock()
_DATE_TOKEN_PATTERN = re.compile(r"%date:([^%]+)%")
_DYNAMIC_TOKEN_PATTERN = re.compile(r"%([^%.]+)\.([^%]+)%")


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _source_type_uri(source_type):
    if source_type in AI_SOURCE_TYPES:
        return AI_SOURCE_TYPES[source_type]
    if source_type in AI_SOURCE_TYPES.values():
        return source_type
    raise ValueError(
        f"Unknown AI source type {source_type!r}. Expected one of: "
        + ", ".join(AI_SOURCE_TYPES)
    )


def _strftime_pattern(comfy_pattern):
    """Translate the date tokens used by ComfyUI's filename widget."""
    replacements = (
        ("yyyy", "%Y"),
        ("yy", "%y"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("HH", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
    )
    translated_pattern = comfy_pattern
    for source_token, strftime_token in replacements:
        translated_pattern = translated_pattern.replace(source_token, strftime_token)
    return translated_pattern


def _prompt_token_values(prompt, extra_pnginfo):
    """Return values addressable as ``%node.widget%`` in a filename prefix."""
    token_values = {}
    workflow_titles = {}
    workflow = (extra_pnginfo or {}).get("workflow") if isinstance(extra_pnginfo, dict) else None
    if isinstance(workflow, dict):
        for workflow_node in workflow.get("nodes", []):
            if not isinstance(workflow_node, dict):
                continue
            node_identifier = str(workflow_node.get("id", ""))
            node_title = workflow_node.get("title")
            if node_identifier and isinstance(node_title, str) and node_title:
                workflow_titles[node_identifier] = node_title
            node_properties = workflow_node.get("properties")
            if node_identifier and isinstance(node_properties, dict):
                search_replace_name = node_properties.get("Node name for S&R")
                if isinstance(search_replace_name, str) and search_replace_name:
                    workflow_titles[node_identifier] = search_replace_name

    if not isinstance(prompt, dict):
        return token_values
    for node_identifier, node_definition in prompt.items():
        if not isinstance(node_definition, dict):
            continue
        input_values = node_definition.get("inputs")
        if not isinstance(input_values, dict):
            continue
        aliases = {str(node_identifier)}
        class_type = node_definition.get("class_type")
        if isinstance(class_type, str) and class_type:
            aliases.add(class_type)
        node_metadata = node_definition.get("_meta")
        if isinstance(node_metadata, dict):
            node_title = node_metadata.get("title")
            if isinstance(node_title, str) and node_title:
                aliases.add(node_title)
        if str(node_identifier) in workflow_titles:
            aliases.add(workflow_titles[str(node_identifier)])
        for alias in aliases:
            for widget_name, widget_value in input_values.items():
                if isinstance(widget_value, (str, int, float, bool)):
                    token_values[(alias, str(widget_name))] = str(widget_value)
    return token_values


def resolve_filename_prefix(
    filename_prefix,
    width=0,
    height=0,
    prompt=None,
    extra_pnginfo=None,
    now=None,
):
    """Resolve Save Image-style date, dimension and node widget tokens."""
    resolved_prefix = str(filename_prefix or "MDPack")
    current_time = now or _datetime.datetime.now()
    resolved_prefix = _DATE_TOKEN_PATTERN.sub(
        lambda match: current_time.strftime(_strftime_pattern(match.group(1))),
        resolved_prefix,
    )
    standard_time_tokens = {
        "%year%": current_time.strftime("%Y"),
        "%month%": current_time.strftime("%m"),
        "%day%": current_time.strftime("%d"),
        "%hour%": current_time.strftime("%H"),
        "%minute%": current_time.strftime("%M"),
        "%second%": current_time.strftime("%S"),
    }
    for token, token_value in standard_time_tokens.items():
        resolved_prefix = resolved_prefix.replace(token, token_value)
    resolved_prefix = resolved_prefix.replace("%width%", str(int(width or 0)))
    resolved_prefix = resolved_prefix.replace("%height%", str(int(height or 0)))

    token_values = _prompt_token_values(prompt, extra_pnginfo)

    def replace_prompt_token(match):
        return token_values.get((match.group(1), match.group(2)), match.group(0))

    return _DYNAMIC_TOKEN_PATTERN.sub(replace_prompt_token, resolved_prefix)


def _safe_prefix_parts(filename_prefix):
    normalized_prefix = os.path.normpath(filename_prefix.replace("\\", "/"))
    if normalized_prefix in ("", "."):
        normalized_prefix = "MDPack"
    prefix_path = Path(normalized_prefix)
    if prefix_path.is_absolute() or ".." in prefix_path.parts:
        raise ValueError("filename_prefix must stay inside ComfyUI's output directory")
    filename_stem = prefix_path.name
    if filename_stem in ("", ".", ".."):
        raise ValueError("filename_prefix must contain a filename")
    return prefix_path.parent, filename_stem


def _open_directory_beneath(parent_directory_fd, directory_name, create_missing):
    """Create/open one real directory without following a symlink."""
    if create_missing:
        try:
            os.mkdir(directory_name, mode=0o755, dir_fd=parent_directory_fd)
        except FileExistsError:
            pass

    try:
        directory_status = os.stat(
            directory_name,
            dir_fd=parent_directory_fd,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ValueError(
            f"filename_prefix directory is not safely accessible: {directory_name}"
        ) from error
    if not stat.S_ISDIR(directory_status.st_mode) or stat.S_ISLNK(directory_status.st_mode):
        raise ValueError(
            f"filename_prefix directory must not be a symlink: {directory_name}"
        )

    directory_open_flags = os.O_RDONLY
    directory_open_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_open_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_open_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        child_directory_fd = os.open(
            directory_name,
            directory_open_flags,
            dir_fd=parent_directory_fd,
        )
    except OSError as error:
        raise ValueError(
            f"filename_prefix directory must not be a symlink: {directory_name}"
        ) from error

    opened_status = os.fstat(child_directory_fd)
    if (
        opened_status.st_dev != directory_status.st_dev
        or opened_status.st_ino != directory_status.st_ino
        or not stat.S_ISDIR(opened_status.st_mode)
    ):
        os.close(child_directory_fd)
        raise ValueError(
            f"filename_prefix directory changed while it was opened: {directory_name}"
        )
    return child_directory_fd


def _open_safe_output_directory(
    output_directory,
    subfolder,
    create_missing=True,
    resolve_output_root=True,
):
    """Return an fd for a descendant reached only through real directories."""
    if resolve_output_root:
        output_root = Path(output_directory).resolve()
    else:
        output_root = Path(os.path.abspath(output_directory))
    if create_missing:
        output_root.mkdir(parents=True, exist_ok=True)
    root_status = os.stat(output_root, follow_symlinks=False)
    if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
        raise ValueError("ComfyUI output directory must not be a symlink")
    root_open_flags = os.O_RDONLY
    root_open_flags |= getattr(os, "O_DIRECTORY", 0)
    root_open_flags |= getattr(os, "O_CLOEXEC", 0)
    root_open_flags |= getattr(os, "O_NOFOLLOW", 0)
    current_directory_fd = os.open(output_root, root_open_flags)
    opened_root_status = os.fstat(current_directory_fd)
    if (
        opened_root_status.st_dev != root_status.st_dev
        or opened_root_status.st_ino != root_status.st_ino
    ):
        os.close(current_directory_fd)
        raise ValueError("ComfyUI output directory changed while it was opened")
    try:
        for directory_name in subfolder.parts:
            if directory_name in ("", "."):
                continue
            child_directory_fd = _open_directory_beneath(
                current_directory_fd,
                directory_name,
                create_missing,
            )
            os.close(current_directory_fd)
            current_directory_fd = child_directory_fd
        root_identity = (opened_root_status.st_dev, opened_root_status.st_ino)
        return output_root, current_directory_fd, root_identity
    except Exception:
        os.close(current_directory_fd)
        raise


class OutputReservation:
    """A reserved output name tied to an open, verified parent directory."""

    def __init__(
        self,
        output_root,
        subfolder,
        output_filename,
        target_directory_fd,
        placeholder_status,
        output_root_identity,
    ):
        self.output_root = output_root
        self.subfolder = subfolder
        self.output_filename = output_filename
        self.target_directory_fd = target_directory_fd
        self.output_root_identity = output_root_identity
        self.owned_file_identity = (
            placeholder_status.st_dev,
            placeholder_status.st_ino,
        )
        self.staging_path = None
        self.closed = False

    @property
    def path(self):
        return self.output_root / self.subfolder / self.output_filename

    @property
    def subfolder_text(self):
        return self.subfolder.as_posix() if str(self.subfolder) != "." else ""

    def create_staging_path(self):
        if self.closed:
            raise RuntimeError("Output reservation is already closed")
        staging_file = tempfile.NamedTemporaryFile(
            prefix="mdpack-stage-",
            suffix=Path(self.output_filename).suffix,
            delete=False,
        )
        staging_file.close()
        self.staging_path = Path(staging_file.name)
        return self.staging_path

    def verify_parent_path(self):
        """Confirm the named parent still resolves to the held directory fd."""
        if self.closed:
            raise RuntimeError("Output reservation is already closed")
        try:
            (
                _output_root,
                verification_directory_fd,
                verification_root_identity,
            ) = _open_safe_output_directory(
                self.output_root,
                self.subfolder,
                create_missing=False,
                resolve_output_root=False,
            )
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "Output directory changed after its filename was reserved"
            ) from error
        try:
            if verification_root_identity != self.output_root_identity:
                raise RuntimeError(
                    "Output root changed after its filename was reserved"
                )
            held_status = os.fstat(self.target_directory_fd)
            verification_status = os.fstat(verification_directory_fd)
            if (
                held_status.st_dev != verification_status.st_dev
                or held_status.st_ino != verification_status.st_ino
            ):
                raise RuntimeError(
                    "Output directory changed after its filename was reserved"
                )
        finally:
            os.close(verification_directory_fd)

    def _verify_owned_file(self):
        try:
            current_status = os.stat(
                self.output_filename,
                dir_fd=self.target_directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("Reserved output placeholder disappeared") from error
        current_identity = (current_status.st_dev, current_status.st_ino)
        if current_identity != self.owned_file_identity:
            raise RuntimeError("Reserved output placeholder was replaced")
        if not stat.S_ISREG(current_status.st_mode) or stat.S_ISLNK(current_status.st_mode):
            raise RuntimeError("Reserved output is not a regular file")

    def publish(self):
        """Copy into the held directory, then atomically publish within it."""
        if self.staging_path is None or not self.staging_path.is_file():
            raise RuntimeError("Completed staging media is missing")
        staging_status = os.stat(self.staging_path, follow_symlinks=False)
        if not stat.S_ISREG(staging_status.st_mode) or stat.S_ISLNK(staging_status.st_mode):
            raise RuntimeError("Completed staging media is not a regular file")
        self.verify_parent_path()
        self._verify_owned_file()
        transient_filename = (
            f".{self.output_filename}.mdpack-{secrets.token_hex(16)}.tmp"
        )
        transient_file_identity = None
        transient_file_fd = None
        try:
            transient_open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            transient_open_flags |= getattr(os, "O_CLOEXEC", 0)
            transient_open_flags |= getattr(os, "O_NOFOLLOW", 0)
            transient_file_fd = os.open(
                transient_filename,
                transient_open_flags,
                0o600,
                dir_fd=self.target_directory_fd,
            )
            transient_status = os.fstat(transient_file_fd)
            transient_file_identity = (
                transient_status.st_dev,
                transient_status.st_ino,
            )
            if not stat.S_ISREG(transient_status.st_mode):
                raise RuntimeError("Transient output is not a regular file")
            with self.staging_path.open("rb") as staging_file:
                while True:
                    media_chunk = staging_file.read(4 * 1024 * 1024)
                    if not media_chunk:
                        break
                    unwritten_chunk = memoryview(media_chunk)
                    while unwritten_chunk:
                        bytes_written = os.write(transient_file_fd, unwritten_chunk)
                        if bytes_written <= 0:
                            raise RuntimeError("Transient output write made no progress")
                        unwritten_chunk = unwritten_chunk[bytes_written:]
            os.fsync(transient_file_fd)
            os.close(transient_file_fd)
            transient_file_fd = None

            copied_status = os.stat(
                transient_filename,
                dir_fd=self.target_directory_fd,
                follow_symlinks=False,
            )
            if (
                (copied_status.st_dev, copied_status.st_ino)
                != transient_file_identity
                or not stat.S_ISREG(copied_status.st_mode)
                or stat.S_ISLNK(copied_status.st_mode)
            ):
                raise RuntimeError("Transient output changed before publish")

            self.verify_parent_path()
            self._verify_owned_file()
            os.replace(
                transient_filename,
                self.output_filename,
                src_dir_fd=self.target_directory_fd,
                dst_dir_fd=self.target_directory_fd,
            )
            os.fsync(self.target_directory_fd)
            self.owned_file_identity = transient_file_identity
            self.verify_parent_path()
            self._verify_owned_file()
        finally:
            if transient_file_fd is not None:
                if transient_file_identity is None:
                    try:
                        transient_status = os.fstat(transient_file_fd)
                        transient_file_identity = (
                            transient_status.st_dev,
                            transient_status.st_ino,
                        )
                    except OSError:
                        pass
                os.close(transient_file_fd)
            if transient_file_identity is not None:
                try:
                    transient_status = os.stat(
                        transient_filename,
                        dir_fd=self.target_directory_fd,
                        follow_symlinks=False,
                    )
                    if (
                        transient_status.st_dev,
                        transient_status.st_ino,
                    ) == transient_file_identity:
                        os.unlink(
                            transient_filename,
                            dir_fd=self.target_directory_fd,
                        )
                except OSError:
                    pass
            if self.staging_path is not None:
                try:
                    self.staging_path.unlink(missing_ok=True)
                except OSError:
                    pass
                self.staging_path = None

    def rollback(self):
        if self.staging_path is not None:
            try:
                self.staging_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.staging_path = None
        if not self.closed:
            try:
                self._verify_owned_file()
                os.unlink(self.output_filename, dir_fd=self.target_directory_fd)
            except (FileNotFoundError, OSError, RuntimeError):
                pass
            finally:
                os.close(self.target_directory_fd)
                self.closed = True

    def validate_success(self):
        if self.closed:
            raise RuntimeError("Output reservation is already closed")
        self.verify_parent_path()
        self._verify_owned_file()

    def release(self):
        if not self.closed:
            os.close(self.target_directory_fd)
            self.closed = True

    def close_success(self):
        self.validate_success()
        self.release()


def reserve_numbered_path(output_directory, filename_prefix, extension):
    """Atomically reserve the next ``prefix_00001_.ext`` output filename."""
    subfolder, filename_stem = _safe_prefix_parts(filename_prefix)
    normalized_extension = extension.lower().lstrip(".")
    filename_pattern = re.compile(
        rf"^{re.escape(filename_stem)}_(\d{{5}})_\.{re.escape(normalized_extension)}$"
    )
    with _COUNTER_LOCK:
        (
            output_root,
            target_directory_fd,
            output_root_identity,
        ) = _open_safe_output_directory(output_directory, subfolder)
        try:
            counters = []
            for existing_filename in os.listdir(target_directory_fd):
                match = filename_pattern.match(existing_filename)
                if match:
                    counters.append(int(match.group(1)))
            counter = max(counters, default=0) + 1
            file_open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            file_open_flags |= getattr(os, "O_CLOEXEC", 0)
            file_open_flags |= getattr(os, "O_NOFOLLOW", 0)
            while True:
                output_filename = (
                    f"{filename_stem}_{counter:05d}_.{normalized_extension}"
                )
                try:
                    file_descriptor = os.open(
                        output_filename,
                        file_open_flags,
                        0o644,
                        dir_fd=target_directory_fd,
                    )
                    placeholder_status = os.fstat(file_descriptor)
                    os.close(file_descriptor)
                    reservation = OutputReservation(
                        output_root,
                        subfolder,
                        output_filename,
                        target_directory_fd,
                        placeholder_status,
                        output_root_identity,
                    )
                    target_directory_fd = None
                    return reservation
                except FileExistsError:
                    counter += 1
        finally:
            if target_directory_fd is not None:
                os.close(target_directory_fd)


def _output_directory():
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.abspath("output")


def _image_arrays(images):
    image_batch = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
    if image_batch.ndim == 3:
        image_batch = image_batch[np.newaxis, ...]
    if image_batch.ndim != 4 or image_batch.shape[-1] not in (1, 3, 4):
        raise ValueError("IMAGE must have shape [batch, height, width, channels]")
    return (np.clip(image_batch, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _workflow_payload(prompt, extra_pnginfo):
    payload = {}
    if prompt is not None:
        payload["prompt"] = prompt
    if isinstance(extra_pnginfo, dict):
        payload.update(extra_pnginfo)
    return payload


def build_xmp_packet(source_type_uri=None, workflow_payload=None):
    """Build XMP carrying IPTC AI provenance and optional ComfyUI JSON."""
    attributes = []
    elements = []
    namespaces = [
        'xmlns:x="adobe:ns:meta/"',
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
    ]
    if source_type_uri:
        namespaces.append(
            'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"'
        )
        attributes.append(
            'Iptc4xmpExt:DigitalSourceType="'
            + html.escape(source_type_uri, quote=True)
            + '"'
        )
    if workflow_payload:
        namespaces.append('xmlns:comfyui="https://comfy.org/ns/workflow/1.0/"')
        workflow_json = html.escape(_json_text(workflow_payload))
        elements.append(f"<comfyui:Workflow>{workflow_json}</comfyui:Workflow>")
    description_attributes = " ".join(['rdf:about=""'] + attributes)
    description_content = "".join(elements)
    return (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        f"<x:xmpmeta {' '.join(namespaces)}>"
        "<rdf:RDF>"
        f"<rdf:Description {description_attributes}>{description_content}</rdf:Description>"
        "</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    ).encode("utf-8")


def _save_image_file(
    image_array,
    output_path,
    image_format,
    quality,
    embed_standalone_ai_xmp,
    source_type_uri,
    embed_workflow,
    prompt,
    extra_pnginfo,
):
    pillow_array = image_array[..., 0] if image_array.shape[-1] == 1 else image_array
    pil_image = Image.fromarray(pillow_array)
    workflow_payload = _workflow_payload(prompt, extra_pnginfo) if embed_workflow else None
    xmp_packet = None
    if embed_standalone_ai_xmp or workflow_payload:
        xmp_packet = build_xmp_packet(
            source_type_uri if embed_standalone_ai_xmp else None,
            workflow_payload,
        )

    normalized_format = image_format.lower()
    save_options = {}
    if normalized_format == "png":
        png_metadata = PngImagePlugin.PngInfo()
        if embed_workflow:
            if prompt is not None:
                png_metadata.add_text("prompt", _json_text(prompt))
            if isinstance(extra_pnginfo, dict):
                for metadata_name, metadata_value in extra_pnginfo.items():
                    png_metadata.add_text(metadata_name, _json_text(metadata_value))
        if xmp_packet:
            png_metadata.add_itxt("XML:com.adobe.xmp", xmp_packet.decode("utf-8"))
        save_options["pnginfo"] = png_metadata
        pillow_format = "PNG"
    elif normalized_format == "jpeg":
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        save_options.update({"quality": int(quality), "subsampling": 0})
        if xmp_packet:
            save_options["xmp"] = xmp_packet
        pillow_format = "JPEG"
    elif normalized_format == "webp":
        save_options["quality"] = int(quality)
        if xmp_packet:
            save_options["xmp"] = xmp_packet
        pillow_format = "WEBP"
    else:
        raise ValueError("Image format must be png, jpeg, or webp")

    temporary_path = output_path.with_name(f".{output_path.name}.encoding")
    try:
        pil_image.save(temporary_path, format=pillow_format, **save_options)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_command(command, tool_label, timeout_seconds, input_bytes=None):
    try:
        completed_process = subprocess.run(
            command,
            input=input_bytes,
            stdin=subprocess.DEVNULL if input_bytes is None else None,
            capture_output=True,
            check=False,
            timeout=float(timeout_seconds),
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{tool_label} was not found. Install it or set its executable path in the node."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{tool_label} timed out after {float(timeout_seconds):g} seconds"
        ) from error
    if completed_process.returncode != 0:
        stderr_text = completed_process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"{tool_label} failed with exit code {completed_process.returncode}: "
            f"{stderr_text[-1200:]}"
        )
    return completed_process


def _validate_signing_inputs(certificate_path, private_key_path):
    if not certificate_path or not private_key_path:
        raise ValueError(
            "C2PA signing requires both a production certificate path and a private key path"
        )
    for label, configured_path in (
        ("C2PA certificate", certificate_path),
        ("C2PA private key", private_key_path),
    ):
        if not Path(configured_path).is_file():
            raise ValueError(f"{label} file does not exist: {configured_path}")


def build_c2pa_manifest(
    media_path,
    source_type_uri,
    manifest_title=None,
):
    return {
        "claim_generator_info": [{"name": "ComfyUI MDPack", "version": "1.0"}],
        "title": manifest_title or media_path.name,
        "format": _MIME_TYPES[media_path.suffix.lower().lstrip(".")],
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": source_type_uri,
                            "softwareAgent": "ComfyUI MDPack",
                        }
                    ],
                    "allActionsIncluded": True,
                },
            },
            {
                "label": "c2pa.metadata",
                "data": {
                    "@context": {
                        "Iptc4xmpExt": "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
                    },
                    "Iptc4xmpExt:DigitalSourceType": source_type_uri,
                },
            },
        ],
    }


def _load_c2pa_sdk():
    try:
        return importlib.import_module("c2pa")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "C2PA signing requires the official c2pa-python SDK. Install this "
            "custom node's requirements (pip install -r requirements.txt)."
        ) from error


def _sign_c2pa(
    media_path,
    source_type_uri,
    certificate_path,
    private_key_path,
    signing_algorithm="es256",
    manifest_title=None,
):
    _validate_signing_inputs(certificate_path, private_key_path)
    c2pa_sdk = _load_c2pa_sdk()
    algorithm_name = str(signing_algorithm).strip().upper()
    try:
        signing_algorithm_value = getattr(c2pa_sdk.C2paSigningAlg, algorithm_name)
    except AttributeError as error:
        raise ValueError(
            f"Unsupported C2PA signing algorithm: {signing_algorithm}"
        ) from error

    signed_path = media_path.with_name(f".{media_path.stem}.signed{media_path.suffix}")
    try:
        certificate_bytes = Path(certificate_path).read_bytes()
        private_key_bytes = Path(private_key_path).read_bytes()
        signer_info = c2pa_sdk.C2paSignerInfo(
            signing_algorithm_value,
            certificate_bytes,
            private_key_bytes,
            None,
        )
        manifest_definition = build_c2pa_manifest(
            media_path,
            source_type_uri,
            manifest_title,
        )
        with c2pa_sdk.Signer.from_info(signer_info) as signer:
            with c2pa_sdk.Builder(manifest_definition) as builder:
                builder.sign_file(media_path, signed_path, signer)
        if not signed_path.is_file() or signed_path.stat().st_size == 0:
            raise RuntimeError("c2pa-python reported success but produced no signed media file")
        os.replace(signed_path, media_path)
    except (ValueError, RuntimeError):
        raise
    except Exception as error:
        raise RuntimeError(f"C2PA signing with c2pa-python failed: {error}") from error
    finally:
        signed_path.unlink(missing_ok=True)


def _probe_video(video_path, ffprobe_path, external_tool_timeout_seconds):
    if not Path(video_path).is_file():
        raise ValueError(f"Video input file does not exist: {video_path}")
    completed_process = _run_command(
        [
            ffprobe_path or "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        "ffprobe",
        external_tool_timeout_seconds,
    )
    try:
        probe_payload = json.loads(completed_process.stdout.decode("utf-8"))
        video_stream = probe_payload["streams"][0]
        return int(video_stream["width"]), int(video_stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("ffprobe found no readable video stream") from error


def _encode_video_from_path(
    source_path,
    output_path,
    ffmpeg_path,
    input_strategy,
    external_tool_timeout_seconds,
):
    command = [
        ffmpeg_path or "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
    ]
    if input_strategy == "remux_all_streams":
        command.extend(
            [
                "-map",
                "0",
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
                "-c",
                "copy",
            ]
        )
    elif input_strategy == "transcode_primary_av":
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
            ]
        )
    else:
        raise ValueError(
            "input_strategy must be remux_all_streams or transcode_primary_av"
        )
    command.extend(["-movflags", "+faststart", str(output_path)])
    _run_command(command, "ffmpeg", external_tool_timeout_seconds)


def _encode_video_from_frames(
    frames,
    fps,
    output_path,
    ffmpeg_path,
    external_tool_timeout_seconds,
):
    if float(fps) <= 0:
        raise ValueError("fps must be greater than zero when encoding IMAGE frames")
    frame_arrays = _image_arrays(frames)
    if frame_arrays.shape[-1] == 4:
        frame_arrays = frame_arrays[..., :3]
    elif frame_arrays.shape[-1] == 1:
        frame_arrays = np.repeat(frame_arrays, 3, axis=-1)
    frame_count, height, width, _channels = frame_arrays.shape
    if frame_count == 0:
        raise ValueError("frames contains no video frames")
    command = [
        ffmpeg_path or "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    stderr_file = tempfile.TemporaryFile()
    ffmpeg_process = None
    writer_thread = None
    try:
        try:
            ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "ffmpeg was not found. Install it or set its executable path in the node."
            ) from error

        writer_errors = []

        def write_frames():
            try:
                for frame_array in frame_arrays:
                    ffmpeg_process.stdin.write(np.ascontiguousarray(frame_array).tobytes())
            except (BrokenPipeError, OSError) as error:
                writer_errors.append(error)
            finally:
                try:
                    ffmpeg_process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

        writer_thread = threading.Thread(target=write_frames, daemon=True)
        writer_thread.start()
        try:
            return_code = ffmpeg_process.wait(
                timeout=float(external_tool_timeout_seconds)
            )
        except subprocess.TimeoutExpired as error:
            ffmpeg_process.kill()
            ffmpeg_process.wait()
            writer_thread.join(timeout=1.0)
            raise RuntimeError(
                "ffmpeg timed out after "
                f"{float(external_tool_timeout_seconds):g} seconds"
            ) from error
        writer_thread.join(timeout=1.0)
        stderr_file.seek(0)
        stderr_text = stderr_file.read().decode("utf-8", errors="replace").strip()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with exit code {return_code}: {stderr_text[-1200:]}"
            )
        if writer_errors:
            raise RuntimeError(f"ffmpeg frame input failed: {writer_errors[0]}")
        if writer_thread.is_alive():
            raise RuntimeError("ffmpeg exited before the frame writer could finish")
    finally:
        if ffmpeg_process is not None and ffmpeg_process.poll() is None:
            ffmpeg_process.kill()
            ffmpeg_process.wait()
        if writer_thread is not None and writer_thread.is_alive():
            writer_thread.join(timeout=1.0)
        stderr_file.close()


def _write_video_xmp(
    video_path,
    embed_standalone_ai_xmp,
    source_type_uri,
    embed_workflow,
    prompt,
    extra_pnginfo,
    exiftool_path,
    external_tool_timeout_seconds,
):
    if not embed_standalone_ai_xmp and not embed_workflow:
        return
    command = [exiftool_path or "exiftool", "-overwrite_original"]
    if embed_standalone_ai_xmp:
        command.append(f"-XMP-iptcExt:DigitalSourceType={source_type_uri}")
    if embed_workflow:
        workflow_description = _json_text(_workflow_payload(prompt, extra_pnginfo))
        command.append(f"-XMP-dc:Description={workflow_description}")
    command.append(str(video_path))
    _run_command(command, "ExifTool", external_tool_timeout_seconds)


class SaveImageWithProvenance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {"default": "MDPack/%date:yyyy-MM-dd%/image"},
                ),
                "format": (["png", "jpeg", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "embed_standalone_ai_xmp": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Write standalone IPTC/XMP AI source metadata. C2PA signing has its own signed AI disclosure and does not depend on this switch.",
                    },
                ),
                "ai_source_type": (list(AI_SOURCE_TYPES),),
                "sign_c2pa": ("BOOLEAN", {"default": False}),
                "certificate_path": ("STRING", {"default": ""}),
                "private_key_path": ("STRING", {"default": ""}),
                "signing_algorithm": (
                    ["es256", "es384", "es512", "ps256", "ps384", "ps512", "ed25519"],
                    {"default": "es256"},
                ),
                "embed_workflow": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Embed the full ComfyUI prompt/workflow. Warning: it may contain API keys, prompts, and local paths.",
                    },
                ),
            },
            "optional": {
                # Accepted for backward-compatible API prompt validation. The
                # web extension removes these obsolete widgets from new graphs.
                "c2patool_path": (
                    "STRING",
                    {"default": "", "tooltip": "Legacy input; ignored."},
                ),
                "tool_timeout_seconds": (
                    "INT",
                    {
                        "default": 300,
                        "min": 1,
                        "max": 86400,
                        "tooltip": "Legacy image-node input; ignored.",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_paths",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "save_images"
    CATEGORY = "MDPack/save"
    OUTPUT_NODE = True

    def save_images(
        self,
        images,
        filename_prefix,
        format,
        quality,
        embed_standalone_ai_xmp,
        ai_source_type,
        sign_c2pa,
        certificate_path,
        private_key_path,
        signing_algorithm,
        embed_workflow,
        prompt=None,
        extra_pnginfo=None,
        c2patool_path=None,
        tool_timeout_seconds=None,
    ):
        image_arrays = _image_arrays(images)
        source_type_uri = _source_type_uri(ai_source_type)
        output_directory = _output_directory()
        saved_paths = []
        ui_images = []
        normalized_format = format.lower()
        reservations = []
        try:
            for batch_number, image_array in enumerate(image_arrays):
                height, width = image_array.shape[:2]
                resolved_prefix = resolve_filename_prefix(
                    filename_prefix,
                    width,
                    height,
                    prompt,
                    extra_pnginfo,
                ).replace("%batch_num%", str(batch_number))
                reservation = reserve_numbered_path(
                    output_directory,
                    resolved_prefix,
                    normalized_format,
                )
                reservations.append(reservation)
                staging_path = reservation.create_staging_path()
                _save_image_file(
                    image_array,
                    staging_path,
                    normalized_format,
                    quality,
                    embed_standalone_ai_xmp,
                    source_type_uri,
                    embed_workflow,
                    prompt,
                    extra_pnginfo,
                )
                if sign_c2pa:
                    _sign_c2pa(
                        staging_path,
                        source_type_uri,
                        certificate_path,
                        private_key_path,
                        signing_algorithm,
                        reservation.output_filename,
                    )
                reservation.publish()
                output_path = reservation.path
                saved_paths.append(str(output_path))
                ui_images.append(
                    {
                        "filename": output_path.name,
                        "subfolder": reservation.subfolder_text,
                        "type": "output",
                    }
                )
            for reservation in reservations:
                reservation.validate_success()
        except Exception:
            for reservation in reservations:
                reservation.rollback()
            raise
        for reservation in reservations:
            reservation.release()
        return {"ui": {"images": ui_images}, "result": (saved_paths,)}


class SaveVideoWithProvenance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": (
                    "STRING",
                    {"default": "MDPack/%date:yyyy-MM-dd%/video"},
                ),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0}),
                "embed_standalone_ai_xmp": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Write standalone IPTC/XMP AI source metadata. C2PA signing has its own signed AI disclosure and does not depend on this switch.",
                    },
                ),
                "ai_source_type": (list(AI_SOURCE_TYPES),),
                "sign_c2pa": ("BOOLEAN", {"default": False}),
                "certificate_path": ("STRING", {"default": ""}),
                "private_key_path": ("STRING", {"default": ""}),
                "signing_algorithm": (
                    ["es256", "es384", "es512", "ps256", "ps384", "ps512", "ed25519"],
                    {"default": "es256"},
                ),
                "tool_timeout_seconds": (
                    "INT",
                    {
                        "default": 300,
                        "min": 1,
                        "max": 86400,
                        "tooltip": "Timeout for FFmpeg, FFprobe, and ExifTool. C2PA signing runs in-process through c2pa-python.",
                    },
                ),
                "embed_workflow": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Embed the full ComfyUI prompt/workflow. Warning: it may contain API keys, prompts, and local paths.",
                    },
                ),
                "ffmpeg_path": ("STRING", {"default": "ffmpeg"}),
                "ffprobe_path": ("STRING", {"default": "ffprobe"}),
                "exiftool_path": ("STRING", {"default": "exiftool"}),
                "input_strategy": (
                    ["remux_all_streams", "transcode_primary_av"],
                    {
                        "default": "remux_all_streams",
                        "tooltip": "Existing video paths: remux every stream without quality loss, or transcode the primary video/audio streams to H.264/AAC for MP4 compatibility.",
                    },
                ),
            },
            "optional": {
                "video_path": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Existing video file. When connected, it takes precedence over frames.",
                    },
                ),
                "frames": ("IMAGE",),
                # Accepted for old API prompts. C2PA now runs in-process, and
                # the web extension removes this obsolete widget from graphs.
                "c2patool_path": (
                    "STRING",
                    {"default": "", "tooltip": "Legacy input; ignored."},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_video"
    CATEGORY = "MDPack/save"
    OUTPUT_NODE = True

    def save_video(
        self,
        filename_prefix,
        fps,
        embed_standalone_ai_xmp,
        ai_source_type,
        sign_c2pa,
        certificate_path,
        private_key_path,
        signing_algorithm,
        tool_timeout_seconds,
        embed_workflow,
        ffmpeg_path,
        ffprobe_path,
        exiftool_path,
        input_strategy,
        video_path=None,
        frames=None,
        prompt=None,
        extra_pnginfo=None,
        c2patool_path=None,
    ):
        source_path = str(video_path or "").strip()
        if source_path:
            width, height = _probe_video(
                source_path,
                ffprobe_path,
                tool_timeout_seconds,
            )
        elif frames is not None:
            frame_arrays = _image_arrays(frames)
            height, width = frame_arrays.shape[1:3]
        else:
            raise ValueError("Connect either video_path or frames")

        source_type_uri = _source_type_uri(ai_source_type)
        resolved_prefix = resolve_filename_prefix(
            filename_prefix,
            width,
            height,
            prompt,
            extra_pnginfo,
        ).replace("%batch_num%", "0")
        reservation = reserve_numbered_path(
            _output_directory(), resolved_prefix, "mp4"
        )
        try:
            staging_path = reservation.create_staging_path()
            if source_path:
                _encode_video_from_path(
                    source_path,
                    staging_path,
                    ffmpeg_path,
                    input_strategy,
                    tool_timeout_seconds,
                )
            else:
                _encode_video_from_frames(
                    frames,
                    fps,
                    staging_path,
                    ffmpeg_path,
                    tool_timeout_seconds,
                )
            _write_video_xmp(
                staging_path,
                embed_standalone_ai_xmp,
                source_type_uri,
                embed_workflow,
                prompt,
                extra_pnginfo,
                exiftool_path,
                tool_timeout_seconds,
            )
            if sign_c2pa:
                _sign_c2pa(
                    staging_path,
                    source_type_uri,
                    certificate_path,
                    private_key_path,
                    signing_algorithm,
                    reservation.output_filename,
                )
            reservation.publish()
            reservation.close_success()
        except Exception:
            reservation.rollback()
            raise
        output_path = reservation.path
        return {
            "ui": {
                "gifs": [
                    {
                        "filename": output_path.name,
                        "subfolder": reservation.subfolder_text,
                        "type": "output",
                    }
                ]
            },
            "result": (str(output_path),),
        }


NODE_CLASS_MAPPINGS = {
    "MDPackSaveImageProvenance": SaveImageWithProvenance,
    "MDPackSaveVideoProvenance": SaveVideoWithProvenance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MDPackSaveImageProvenance": "Save Image (AI Metadata + C2PA)",
    "MDPackSaveVideoProvenance": "Save Video (AI Metadata + C2PA)",
}
