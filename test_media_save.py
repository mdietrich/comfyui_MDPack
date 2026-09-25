"""Focused tests for provenance-aware media save nodes."""

import datetime
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import media_save


class FolderPathsStub:
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def get_output_directory(self):
        return self.output_directory


class MediaSaveTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_directory = Path(self.temporary_directory.name)
        self.folder_paths_patch = mock.patch.object(
            media_save,
            "folder_paths",
            FolderPathsStub(str(self.output_directory)),
        )
        self.folder_paths_patch.start()

    def tearDown(self):
        self.folder_paths_patch.stop()
        self.temporary_directory.cleanup()

    def test_dynamic_prefix_and_atomic_numbering(self):
        prompt = {
            "12": {
                "class_type": "KSampler",
                "_meta": {"title": "Main sampler"},
                "inputs": {"seed": 42, "complex": ["not", "scalar"]},
            }
        }
        resolved_prefix = media_save.resolve_filename_prefix(
            "renders/%date:yyyy-MM-dd%/%Main sampler.seed%_%width%x%height%_"
            "%year%%month%%day%-%hour%%minute%%second%",
            width=1024,
            height=768,
            prompt=prompt,
            now=datetime.datetime(2026, 8, 22, 13, 14, 15),
        )
        self.assertEqual(
            resolved_prefix,
            "renders/2026-08-22/42_1024x768_20260822-131415",
        )

        first_reservation = media_save.reserve_numbered_path(
            self.output_directory, resolved_prefix, "png"
        )
        second_reservation = media_save.reserve_numbered_path(
            self.output_directory, resolved_prefix, "png"
        )
        self.assertEqual(
            first_reservation.path.name,
            "42_1024x768_20260822-131415_00001_.png",
        )
        self.assertEqual(
            second_reservation.path.name,
            "42_1024x768_20260822-131415_00002_.png",
        )
        self.assertEqual(first_reservation.subfolder_text, "renders/2026-08-22")
        self.assertEqual(
            second_reservation.subfolder_text,
            first_reservation.subfolder_text,
        )
        first_reservation.rollback()
        second_reservation.rollback()

    def test_controlled_ai_source_types_use_exact_iptc_uris(self):
        self.assertEqual(
            media_save.AI_SOURCE_TYPES["Fully AI-generated"],
            "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
        )
        self.assertEqual(
            media_save.AI_SOURCE_TYPES["AI-edited / composite"],
            "http://cv.iptc.org/newscodes/digitalsourcetype/"
            "compositeWithTrainedAlgorithmicMedia",
        )

    def test_prefix_cannot_escape_output_directory(self):
        with self.assertRaisesRegex(ValueError, "inside"):
            media_save.reserve_numbered_path(self.output_directory, "../escape", "png")

    def test_symlinked_output_subfolder_cannot_create_outside_directories(self):
        with tempfile.TemporaryDirectory() as outside_directory_name:
            outside_directory = Path(outside_directory_name)
            symlink_path = self.output_directory / "jump"
            symlink_path.symlink_to(outside_directory, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symlink"):
                media_save.reserve_numbered_path(
                    self.output_directory,
                    "jump/new/file",
                    "png",
                )

            self.assertFalse((outside_directory / "new").exists())
            self.assertEqual(list(outside_directory.iterdir()), [])

    def test_parent_swap_after_reservation_fails_closed_without_outside_write(self):
        node = media_save.SaveImageWithProvenance()
        with tempfile.TemporaryDirectory() as outside_directory_name:
            outside_directory = Path(outside_directory_name)
            target_parent = self.output_directory / "swap"
            moved_parent = outside_directory / "moved-parent"
            staging_paths = []

            def swap_parent_then_write_staging(image_array, staging_path, *args):
                staging_paths.append(Path(staging_path))
                self.assertEqual(
                    Path(staging_path).parent.resolve(),
                    Path(tempfile.gettempdir()).resolve(),
                )
                target_parent.rename(moved_parent)
                target_parent.symlink_to(outside_directory, target_is_directory=True)
                Path(staging_path).write_bytes(b"completed-staging-image")

            with mock.patch.object(
                media_save,
                "_save_image_file",
                side_effect=swap_parent_then_write_staging,
            ):
                with self.assertRaisesRegex(RuntimeError, "Output directory changed"):
                    node.save_images(
                        images=np.zeros((1, 2, 2, 3), dtype=np.float32),
                        filename_prefix="swap/image",
                        format="png",
                        quality=95,
                        embed_standalone_ai_xmp=False,
                        ai_source_type="Fully AI-generated",
                        sign_c2pa=False,
                        certificate_path="",
                        private_key_path="",
                        signing_algorithm="es256",
                        embed_workflow=False,
                    )

            self.assertEqual(
                [path for path in outside_directory.rglob("*") if path.is_file()],
                [],
            )
            self.assertEqual(
                [path for path in staging_paths if path.exists()],
                [],
            )

    def test_root_swap_before_staging_fails_closed_without_outside_write(self):
        node = media_save.SaveImageWithProvenance()
        configured_root = self.output_directory / "configured-output"
        configured_root.mkdir()
        original_create_staging_path = media_save.OutputReservation.create_staging_path
        staging_paths = []

        with tempfile.TemporaryDirectory() as outside_directory_name:
            outside_directory = Path(outside_directory_name)
            moved_root = outside_directory / "moved-root"

            def swap_root_then_create_staging(reservation):
                configured_root.rename(moved_root)
                configured_root.symlink_to(outside_directory, target_is_directory=True)
                staging_path = original_create_staging_path(reservation)
                staging_paths.append(staging_path)
                return staging_path

            def write_staging(image_array, staging_path, *args):
                Path(staging_path).write_bytes(b"completed-staging-image")

            with mock.patch.object(
                media_save,
                "folder_paths",
                FolderPathsStub(str(configured_root)),
            ), mock.patch.object(
                media_save.OutputReservation,
                "create_staging_path",
                autospec=True,
                side_effect=swap_root_then_create_staging,
            ), mock.patch.object(
                media_save,
                "_save_image_file",
                side_effect=write_staging,
            ):
                with self.assertRaisesRegex(RuntimeError, "Output directory changed"):
                    node.save_images(
                        images=np.zeros((1, 2, 2, 3), dtype=np.float32),
                        filename_prefix="nested/image",
                        format="png",
                        quality=95,
                        embed_standalone_ai_xmp=False,
                        ai_source_type="Fully AI-generated",
                        sign_c2pa=False,
                        certificate_path="",
                        private_key_path="",
                        signing_algorithm="es256",
                        embed_workflow=False,
                    )

            self.assertEqual(
                [path for path in outside_directory.rglob("*") if path.is_file()],
                [],
            )
            self.assertEqual([path for path in staging_paths if path.exists()], [])
            configured_root.unlink()

    def test_png_metadata_switches_are_independent(self):
        output_path = self.output_directory / "metadata.png"
        image_array = np.zeros((4, 5, 3), dtype=np.uint8)
        source_type_uri = media_save.AI_SOURCE_TYPES["Fully AI-generated"]
        prompt = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
        extra_pnginfo = {"workflow": {"nodes": [{"id": 1}]}}

        media_save._save_image_file(
            image_array,
            output_path,
            "png",
            95,
            True,
            source_type_uri,
            False,
            prompt,
            extra_pnginfo,
        )
        with Image.open(output_path) as saved_image:
            self.assertNotIn("prompt", saved_image.info)
            self.assertNotIn("workflow", saved_image.info)
            self.assertIn(source_type_uri, saved_image.info["XML:com.adobe.xmp"])

        media_save._save_image_file(
            image_array,
            output_path,
            "png",
            95,
            False,
            source_type_uri,
            True,
            prompt,
            extra_pnginfo,
        )
        with Image.open(output_path) as saved_image:
            self.assertEqual(json.loads(saved_image.info["prompt"]), prompt)
            self.assertEqual(json.loads(saved_image.info["workflow"]), extra_pnginfo["workflow"])
            self.assertNotIn(source_type_uri, saved_image.info["XML:com.adobe.xmp"])

    def test_workflow_embedding_defaults_off_to_avoid_secret_leaks(self):
        image_inputs = media_save.SaveImageWithProvenance.INPUT_TYPES()["required"]
        video_inputs = media_save.SaveVideoWithProvenance.INPUT_TYPES()["required"]
        image_optional = media_save.SaveImageWithProvenance.INPUT_TYPES()["optional"]
        video_optional = media_save.SaveVideoWithProvenance.INPUT_TYPES()["optional"]
        self.assertIn("embed_standalone_ai_xmp", image_inputs)
        self.assertIn("embed_standalone_ai_xmp", video_inputs)
        self.assertNotIn("embed_ai_metadata", image_inputs)
        self.assertNotIn("embed_ai_metadata", video_inputs)
        self.assertIn("c2patool_path", image_optional)
        self.assertIn("c2patool_path", video_optional)
        self.assertIn("tool_timeout_seconds", image_optional)
        self.assertIn("tool_timeout_seconds", video_inputs)
        self.assertNotIn("external_tool_timeout_seconds", image_inputs)
        self.assertNotIn("external_tool_timeout_seconds", video_inputs)
        self.assertFalse(image_inputs["embed_workflow"][1]["default"])
        self.assertFalse(video_inputs["embed_workflow"][1]["default"])
        self.assertIn("API keys", image_inputs["embed_workflow"][1]["tooltip"])

    def test_legacy_image_api_inputs_are_accepted_and_ignored(self):
        node = media_save.SaveImageWithProvenance()
        result = node.save_images(
            images=np.zeros((1, 2, 3, 3), dtype=np.float32),
            filename_prefix="legacy/api_image",
            format="png",
            quality=95,
            embed_standalone_ai_xmp=False,
            ai_source_type="Fully AI-generated",
            sign_c2pa=False,
            certificate_path="",
            private_key_path="",
            signing_algorithm="es256",
            embed_workflow=False,
            c2patool_path="/obsolete/c2patool",
            tool_timeout_seconds=17,
        )
        saved_path = Path(result["result"][0][0])
        self.assertTrue(saved_path.is_file())
        with Image.open(saved_path) as saved_image:
            self.assertNotIn("prompt", saved_image.info)
            self.assertNotIn("workflow", saved_image.info)

    def test_all_image_formats_are_written(self):
        image_array = np.zeros((4, 6, 4), dtype=np.uint8)
        for image_format in ("png", "jpeg", "webp"):
            with self.subTest(image_format=image_format):
                output_path = self.output_directory / f"image.{image_format}"
                media_save._save_image_file(
                    image_array,
                    output_path,
                    image_format,
                    90,
                    False,
                    media_save.AI_SOURCE_TYPES["Fully AI-generated"],
                    False,
                    None,
                    None,
                )
                self.assertGreater(output_path.stat().st_size, 0)
                with Image.open(output_path) as saved_image:
                    self.assertEqual(saved_image.size, (6, 4))

    def test_single_channel_image_is_saved_as_grayscale(self):
        output_path = self.output_directory / "grayscale.png"
        image_array = np.zeros((4, 6, 1), dtype=np.uint8)
        media_save._save_image_file(
            image_array,
            output_path,
            "png",
            90,
            False,
            media_save.AI_SOURCE_TYPES["Fully AI-generated"],
            False,
            None,
            None,
        )
        with Image.open(output_path) as saved_image:
            self.assertEqual(saved_image.mode, "L")
            self.assertEqual(saved_image.size, (6, 4))

    def test_c2pa_manifest_uses_normative_created_action(self):
        source_type_uri = media_save.AI_SOURCE_TYPES["Fully AI-generated"]
        media_path = self.output_directory / "source.jpg"
        manifest = media_save.build_c2pa_manifest(
            media_path,
            source_type_uri,
            "final-output.jpg",
        )

        self.assertEqual(manifest["title"], "final-output.jpg")
        self.assertEqual(manifest["format"], "image/jpeg")
        self.assertEqual(
            manifest["claim_generator_info"],
            [{"name": "ComfyUI MDPack", "version": "1.0"}],
        )
        assertions = manifest["assertions"]
        self.assertEqual(assertions[0]["label"], "c2pa.actions.v2")
        first_action = assertions[0]["data"]["actions"][0]
        self.assertEqual(first_action["action"], "c2pa.created")
        self.assertEqual(first_action["digitalSourceType"], source_type_uri)
        self.assertNotIn("c2pa.actions.gen-ai", [item["label"] for item in assertions])
        self.assertEqual(
            assertions[1]["data"]["Iptc4xmpExt:DigitalSourceType"],
            source_type_uri,
        )

    def test_c2pa_sdk_import_error_has_installation_hint(self):
        with mock.patch.object(
            media_save.importlib,
            "import_module",
            side_effect=ImportError("not installed"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "c2pa-python SDK.*pip install -r requirements.txt",
            ):
                media_save._load_c2pa_sdk()

    def test_real_c2pa_sdk_signs_and_reads_jpeg(self):
        certificate_path = os.environ.get("MDPACK_C2PA_TEST_CERT")
        private_key_path = os.environ.get("MDPACK_C2PA_TEST_KEY")
        if not certificate_path or not private_key_path:
            self.skipTest("Set MDPACK_C2PA_TEST_CERT and MDPACK_C2PA_TEST_KEY")
        try:
            import c2pa
        except ImportError:
            self.skipTest("c2pa-python is not installed in the test environment")

        source_type_uri = media_save.AI_SOURCE_TYPES["Fully AI-generated"]
        media_path = self.output_directory / "sdk-signed.jpg"
        Image.new("RGB", (8, 6), color=(10, 20, 30)).save(media_path, format="JPEG")

        media_save._sign_c2pa(
            media_path,
            source_type_uri,
            certificate_path,
            private_key_path,
            "es256",
            "final-sdk-output.jpg",
        )

        with c2pa.Reader(media_path) as reader:
            manifest = reader.get_active_manifest()
            validation_state = reader.get_validation_state()
        self.assertEqual(str(validation_state), "Valid")
        self.assertEqual(manifest["title"], "final-sdk-output.jpg")
        assertions = manifest["assertions"]
        actions_assertion = next(
            assertion
            for assertion in assertions
            if assertion["label"].startswith("c2pa.actions.v2")
        )
        first_action = actions_assertion["data"]["actions"][0]
        self.assertEqual(first_action["action"], "c2pa.created")
        self.assertEqual(first_action["digitalSourceType"], source_type_uri)
        metadata_assertion = next(
            assertion
            for assertion in assertions
            if assertion["label"].startswith("c2pa.metadata")
        )
        self.assertEqual(
            metadata_assertion["data"]["Iptc4xmpExt:DigitalSourceType"],
            source_type_uri,
        )

    def test_requested_signing_failure_removes_unsigned_image(self):
        node = media_save.SaveImageWithProvenance()
        image_batch = np.zeros((1, 3, 4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "production certificate"):
            node.save_images(
                images=image_batch,
                filename_prefix="signed/image",
                format="png",
                quality=95,
                embed_standalone_ai_xmp=True,
                ai_source_type="Fully AI-generated",
                sign_c2pa=True,
                certificate_path="",
                private_key_path="",
                signing_algorithm="es256",
                embed_workflow=False,
            )
        self.assertEqual(list(self.output_directory.rglob("*.png")), [])

    def test_c2pa_ai_disclosure_is_independent_of_standalone_xmp(self):
        node = media_save.SaveImageWithProvenance()
        sign_arguments = []

        def fake_save(image_array, output_path, *args):
            self.assertEqual(
                Path(output_path).parent.resolve(),
                Path(tempfile.gettempdir()).resolve(),
            )
            self.assertTrue(Path(output_path).name.startswith("mdpack-stage-"))
            Path(output_path).write_bytes(b"encoded")

        def fake_sign(*args):
            sign_arguments.append(args)

        with mock.patch.object(media_save, "_save_image_file", side_effect=fake_save), mock.patch.object(
            media_save, "_sign_c2pa", side_effect=fake_sign
        ):
            node.save_images(
                images=np.zeros((1, 2, 2, 3), dtype=np.float32),
                filename_prefix="signed/no_standalone_xmp",
                format="png",
                quality=95,
                embed_standalone_ai_xmp=False,
                ai_source_type="Fully AI-generated",
                sign_c2pa=True,
                certificate_path="certificate.pem",
                private_key_path="private.key",
                signing_algorithm="es256",
                embed_workflow=False,
            )
        self.assertEqual(len(sign_arguments), 1)
        self.assertEqual(
            sign_arguments[0][1],
            media_save.AI_SOURCE_TYPES["Fully AI-generated"],
        )
        self.assertFalse(Path(sign_arguments[0][0]).exists())
        self.assertEqual(
            list(self.output_directory.rglob(".*.mdpack-*.tmp")),
            [],
        )

    def test_batch_failure_removes_every_image_created_by_the_call(self):
        node = media_save.SaveImageWithProvenance()
        image_batch = np.zeros((3, 3, 4, 3), dtype=np.float32)
        save_count = 0

        def fail_on_second_image(image_array, output_path, *args):
            nonlocal save_count
            save_count += 1
            Path(output_path).write_bytes(b"partial")
            if save_count == 2:
                raise RuntimeError("second image failed")

        with mock.patch.object(
            media_save, "_save_image_file", side_effect=fail_on_second_image
        ):
            with self.assertRaisesRegex(RuntimeError, "second image failed"):
                node.save_images(
                    images=image_batch,
                    filename_prefix="batch/image_%batch_num%",
                    format="png",
                    quality=95,
                    embed_standalone_ai_xmp=False,
                    ai_source_type="Fully AI-generated",
                    sign_c2pa=False,
                    certificate_path="",
                    private_key_path="",
                    signing_algorithm="es256",
                    embed_workflow=False,
                )
        self.assertEqual(list(self.output_directory.rglob("*.png")), [])

    def test_existing_video_strategies_are_explicit_and_lossless_by_default(self):
        video_inputs = media_save.SaveVideoWithProvenance.INPUT_TYPES()["required"]
        self.assertEqual(video_inputs["input_strategy"][1]["default"], "remux_all_streams")
        captured_commands = []

        def capture_command(command, tool_label, timeout_seconds, input_bytes=None):
            captured_commands.append((command, tool_label, timeout_seconds))

        with mock.patch.object(media_save, "_run_command", side_effect=capture_command):
            media_save._encode_video_from_path(
                "input.mov", "output.mp4", "ffmpeg", "remux_all_streams", 19
            )
            media_save._encode_video_from_path(
                "input.mov", "output.mp4", "ffmpeg", "transcode_primary_av", 23
            )

        remux_command, remux_label, remux_timeout = captured_commands[0]
        self.assertEqual(remux_label, "ffmpeg")
        self.assertEqual(remux_timeout, 19)
        self.assertIn("-nostdin", remux_command)
        self.assertIn(["-map", "0"], [remux_command[index:index + 2] for index in range(len(remux_command) - 1)])
        self.assertIn(["-map_metadata", "0"], [remux_command[index:index + 2] for index in range(len(remux_command) - 1)])
        self.assertIn(["-map_chapters", "0"], [remux_command[index:index + 2] for index in range(len(remux_command) - 1)])
        self.assertIn(["-c", "copy"], [remux_command[index:index + 2] for index in range(len(remux_command) - 1)])
        self.assertNotIn("libx264", remux_command)

        transcode_command, _label, transcode_timeout = captured_commands[1]
        self.assertEqual(transcode_timeout, 23)
        self.assertIn("libx264", transcode_command)
        self.assertIn("aac", transcode_command)
        self.assertIn("pad=ceil(iw/2)*2:ceil(ih/2)*2", transcode_command)

    def test_publish_copies_large_staging_media_in_bounded_chunks(self):
        reservation = media_save.reserve_numbered_path(
            self.output_directory,
            "large/media",
            "bin",
        )
        staging_path = reservation.create_staging_path()
        media_size = 9 * 1024 * 1024 + 7
        with staging_path.open("wb") as staging_file:
            staging_file.seek(media_size - 1)
            staging_file.write(b"x")

        original_os_write = media_save.os.write
        write_sizes = []

        def recording_write(file_descriptor, data):
            write_sizes.append(len(data))
            return original_os_write(file_descriptor, data)

        try:
            with mock.patch.object(media_save.os, "write", side_effect=recording_write):
                reservation.publish()
            reservation.close_success()
        except Exception:
            reservation.rollback()
            raise

        self.assertGreaterEqual(len(write_sizes), 3)
        self.assertLessEqual(max(write_sizes), 4 * 1024 * 1024)
        self.assertEqual(reservation.path.stat().st_size, media_size)
        self.assertFalse(staging_path.exists())
        self.assertEqual(
            list(self.output_directory.rglob(".*.mdpack-*.tmp")),
            [],
        )

    def test_frame_encoding_streams_one_frame_at_a_time_and_pads_odd_dimensions(self):
        written_chunks = []
        process_holder = {}

        class FakeStdin:
            def write(self, chunk):
                written_chunks.append(chunk)

            def close(self):
                return None

        class FakeProcess:
            def __init__(self):
                self.stdin = FakeStdin()
                self.return_code = None
                self.killed = False

            def wait(self, timeout=None):
                self.return_code = 0
                return 0

            def poll(self):
                return self.return_code

            def kill(self):
                self.killed = True
                self.return_code = -9

        def fake_popen(command, **kwargs):
            process_holder["command"] = command
            process_holder["kwargs"] = kwargs
            process_holder["process"] = FakeProcess()
            return process_holder["process"]

        frames = np.zeros((3, 3, 5, 3), dtype=np.float32)
        with mock.patch("subprocess.Popen", side_effect=fake_popen):
            media_save._encode_video_from_frames(frames, 24, "output.mp4", "ffmpeg", 20)

        self.assertEqual(len(written_chunks), 3)
        self.assertTrue(all(len(chunk) == 3 * 5 * 3 for chunk in written_chunks))
        self.assertIn("-nostdin", process_holder["command"])
        self.assertIn("pad=ceil(iw/2)*2:ceil(ih/2)*2", process_holder["command"])
        self.assertIs(process_holder["kwargs"]["stdout"], media_save.subprocess.DEVNULL)

    def test_frame_encoding_timeout_kills_process(self):
        class FakeStdin:
            def write(self, chunk):
                return len(chunk)

            def close(self):
                return None

        class TimeoutProcess:
            def __init__(self):
                self.stdin = FakeStdin()
                self.return_code = None
                self.killed = False

            def wait(self, timeout=None):
                if timeout is not None and not self.killed:
                    raise media_save.subprocess.TimeoutExpired("ffmpeg", timeout)
                self.return_code = -9
                return self.return_code

            def poll(self):
                return self.return_code

            def kill(self):
                self.killed = True

        fake_process = TimeoutProcess()
        with mock.patch("subprocess.Popen", return_value=fake_process):
            with self.assertRaisesRegex(RuntimeError, "timed out after 2 seconds"):
                media_save._encode_video_from_frames(
                    np.zeros((1, 2, 2, 3), dtype=np.float32),
                    24,
                    "output.mp4",
                    "ffmpeg",
                    2,
                )
        self.assertTrue(fake_process.killed)

    def test_video_steps_run_encode_then_xmp_then_c2pa(self):
        node = media_save.SaveVideoWithProvenance()
        source_path = self.output_directory / "source.mov"
        source_path.write_bytes(b"source")
        call_order = []
        processed_paths = []

        def fake_probe(path, executable, timeout_seconds):
            call_order.append("probe")
            return 640, 480

        def fake_encode(path, output_path, executable, strategy, timeout_seconds):
            call_order.append("encode")
            processed_paths.append(Path(output_path))
            Path(output_path).write_bytes(b"encoded")

        def fake_xmp(*args):
            call_order.append("xmp")
            processed_paths.append(Path(args[0]))

        def fake_sign(*args):
            call_order.append("c2pa")
            processed_paths.append(Path(args[0]))

        with mock.patch.object(media_save, "_probe_video", side_effect=fake_probe), mock.patch.object(
            media_save, "_encode_video_from_path", side_effect=fake_encode
        ), mock.patch.object(media_save, "_write_video_xmp", side_effect=fake_xmp), mock.patch.object(
            media_save, "_sign_c2pa", side_effect=fake_sign
        ):
            result = node.save_video(
                filename_prefix="videos/render",
                fps=24.0,
                embed_standalone_ai_xmp=True,
                ai_source_type="Fully AI-generated",
                sign_c2pa=True,
                certificate_path="certificate.pem",
                private_key_path="private.key",
                signing_algorithm="es256",
                tool_timeout_seconds=300,
                embed_workflow=True,
                ffmpeg_path="ffmpeg",
                ffprobe_path="ffprobe",
                exiftool_path="exiftool",
                input_strategy="remux_all_streams",
                video_path=str(source_path),
                prompt={"1": {}},
                extra_pnginfo={"workflow": {}},
                c2patool_path="/obsolete/c2patool",
            )
        self.assertEqual(call_order, ["probe", "encode", "xmp", "c2pa"])
        self.assertEqual(len(set(processed_paths)), 1)
        self.assertEqual(
            processed_paths[0].parent.resolve(),
            Path(tempfile.gettempdir()).resolve(),
        )
        self.assertTrue(processed_paths[0].name.startswith("mdpack-stage-"))
        self.assertFalse(processed_paths[0].exists())
        self.assertEqual(
            list(self.output_directory.rglob(".*.mdpack-*.tmp")),
            [],
        )
        self.assertTrue(Path(result["result"][0]).is_file())

    def test_video_xmp_switches_control_exiftool_arguments(self):
        video_path = self.output_directory / "video.mp4"
        video_path.write_bytes(b"video")
        commands = []

        def capture_command(command, tool_label, timeout_seconds, input_bytes=None):
            commands.append(command)

        with mock.patch.object(media_save, "_run_command", side_effect=capture_command):
            media_save._write_video_xmp(
                video_path,
                False,
                media_save.AI_SOURCE_TYPES["Fully AI-generated"],
                False,
                {"prompt": True},
                {"workflow": {}},
                "exiftool",
                300,
            )
            self.assertEqual(commands, [])

            media_save._write_video_xmp(
                video_path,
                True,
                media_save.AI_SOURCE_TYPES["AI-edited / composite"],
                False,
                None,
                None,
                "exiftool",
                300,
            )
            self.assertTrue(any("DigitalSourceType=" in argument for argument in commands[-1]))
            self.assertFalse(any("Description=" in argument for argument in commands[-1]))

            media_save._write_video_xmp(
                video_path,
                False,
                media_save.AI_SOURCE_TYPES["Fully AI-generated"],
                True,
                {"prompt": True},
                {"workflow": {"nodes": []}},
                "exiftool",
                300,
            )
            self.assertFalse(any("DigitalSourceType=" in argument for argument in commands[-1]))
            self.assertTrue(any("Description=" in argument for argument in commands[-1]))

    def test_external_tool_failure_is_clear(self):
        completed_process = mock.Mock(returncode=2, stderr=b"bad input", stdout=b"")
        with mock.patch("subprocess.run", return_value=completed_process) as run_mock:
            with self.assertRaisesRegex(RuntimeError, "ffmpeg failed.*bad input"):
                media_save._run_command(["ffmpeg", "-i", "bad"], "ffmpeg", 12)
        self.assertIs(run_mock.call_args.kwargs["stdin"], media_save.subprocess.DEVNULL)
        self.assertEqual(run_mock.call_args.kwargs["timeout"], 12.0)

    def test_external_tool_timeout_is_clear(self):
        timeout_error = media_save.subprocess.TimeoutExpired(["ffprobe"], 7)
        with mock.patch("subprocess.run", side_effect=timeout_error):
            with self.assertRaisesRegex(RuntimeError, "ffprobe timed out after 7 seconds"):
                media_save._run_command(["ffprobe", "input.mp4"], "ffprobe", 7)


if __name__ == "__main__":
    unittest.main()
