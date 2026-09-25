"""Tests for the Atlas Cloud HTTP routes.

Run remotely:
    ~/ComfyUI_winows_portable/.venv/bin/python test_atlascloud_routes.py

The ComfyUI host was offline when this suite was written, so route behaviour
is verified by calling the handlers directly with a fake aiohttp request
instead of curling a running server (see the task-5 report for details).
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlascloud_models as am  # noqa: E402
import atlascloud_routes as routes  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"PASS {name}")
    else:
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


class FakeRequest:
    """Just enough of an aiohttp request for the handlers under test."""

    def __init__(self, query=None, headers=None):
        self.query = query or {}
        self.headers = headers or {}


def run(coroutine):
    return asyncio.run(coroutine)


def body_json(response):
    return json.loads(response.body.decode("utf-8"))


SAMPLE_MODELS = [
    {"model": "bytedance/seedream-v5.0-pro/text-to-image", "display_name": "Seedream v5.0 Pro",
     "category": "TEXT-TO-IMAGE", "input_kind": "Text", "output_kind": "Image",
     "schema_url": "https://static.test/seedream-t2i.json", "price": "0.036"},
    {"model": "bytedance/seedream-v5.0-pro/edit", "display_name": "Seedream v5.0 Pro Edit",
     "category": "IMAGE-TO-IMAGE", "input_kind": "Image", "output_kind": "Image",
     "schema_url": "https://static.test/seedream-edit.json", "price": "0.036"},
]

SAMPLE_SCHEMA = {
    "model": "bytedance/seedream-v5.0-pro/edit",
    "prompt_field": "prompt",
    "image_fields": [{"name": "images", "is_list": True, "max_items": 10}],
    "fields": [{"name": "size", "type": "string", "default": None, "enum": None,
                "required": False, "description": "", "is_media": False, "is_list": False}],
}

# --- handle_models: no filters ----------------------------------------------
list_models_calls = []


def fake_list_models(output_kind="", input_kind="", **load_kwargs):
    list_models_calls.append({"output_kind": output_kind, "input_kind": input_kind, **load_kwargs})
    return list(SAMPLE_MODELS)


am.list_models = fake_list_models

response = run(routes.handle_models(FakeRequest()))
payload = body_json(response)
check("handle_models returns every model with no filters",
      response.status == 200 and payload["models"] == SAMPLE_MODELS, str(payload))
check("handle_models returns the output kinds list", payload["output_kinds"] == am.OUTPUT_KINDS)
check("handle_models returns the input kinds list", payload["input_kinds"] == am.INPUT_KINDS)
check("handle_models called list_models with empty filters and no forced refresh",
      list_models_calls[-1] == {"output_kind": "", "input_kind": "", "force_refresh": False,
                                "provider": "atlascloud", "api_key": ""},
      str(list_models_calls[-1]))

# --- handle_models: filters and refresh pass-through ------------------------
list_models_calls.clear()
run(routes.handle_models(FakeRequest({"output": "Image", "input": "Image"})))
check("handle_models passes the output and input query parameters through",
      list_models_calls[-1] == {"output_kind": "Image", "input_kind": "Image", "force_refresh": False,
                                "provider": "atlascloud", "api_key": ""},
      str(list_models_calls[-1]))

list_models_calls.clear()
run(routes.handle_models(FakeRequest({"refresh": "1"})))
check("handle_models maps refresh=1 to force_refresh=True",
      list_models_calls[-1]["force_refresh"] is True, str(list_models_calls[-1]))

list_models_calls.clear()
run(routes.handle_models(FakeRequest({"refresh": "0"})))
check("handle_models maps refresh=0 to force_refresh=False",
      list_models_calls[-1]["force_refresh"] is False, str(list_models_calls[-1]))

# --- handle_models: upstream failure -----------------------------------------
def failing_list_models(**kwargs):
    raise RuntimeError("catalogue unreachable")


am.list_models = failing_list_models
error_response = run(routes.handle_models(FakeRequest()))
error_payload = body_json(error_response)
check("handle_models returns HTTP 502 when list_models raises", error_response.status == 502)
check("handle_models error payload carries the error message and an empty models list",
      error_payload.get("error") == "catalogue unreachable" and error_payload.get("models") == [],
      str(error_payload))

am.list_models = fake_list_models

# --- handle_schema ------------------------------------------------------------
schema_calls = []


def fake_schema_for_model(model_identifier, **kwargs):
    schema_calls.append(model_identifier)
    return dict(SAMPLE_SCHEMA)


am.schema_for_model = fake_schema_for_model

schema_response = run(routes.handle_schema(FakeRequest({"model": "bytedance/seedream-v5.0-pro/edit"})))
schema_payload = body_json(schema_response)
check("handle_schema returns the normalised schema for the requested model",
      schema_response.status == 200 and schema_payload == SAMPLE_SCHEMA, str(schema_payload))
check("handle_schema passed the model identifier through",
      schema_calls[-1] == "bytedance/seedream-v5.0-pro/edit")

missing_model_response = run(routes.handle_schema(FakeRequest()))
missing_model_payload = body_json(missing_model_response)
check("handle_schema returns HTTP 400 when model is missing", missing_model_response.status == 400)
check("handle_schema 400 payload carries an error key", "error" in missing_model_payload,
      str(missing_model_payload))


def failing_schema_for_model(model_identifier, **kwargs):
    raise RuntimeError("schema unreachable")


am.schema_for_model = failing_schema_for_model
schema_error_response = run(routes.handle_schema(FakeRequest({"model": "some/model"})))
schema_error_payload = body_json(schema_error_response)
check("handle_schema returns HTTP 502 when schema_for_model raises",
      schema_error_response.status == 502)
check("handle_schema 502 payload carries the error message",
      schema_error_payload.get("error") == "schema unreachable", str(schema_error_payload))

am.schema_for_model = fake_schema_for_model

# --- handle_refresh -----------------------------------------------------------
memo_cleared = []
original_clear_schema_memo = am.clear_schema_memo


def tracking_clear_schema_memo():
    memo_cleared.append(True)
    original_clear_schema_memo()


am.clear_schema_memo = tracking_clear_schema_memo
am.list_models = fake_list_models

refresh_response = run(routes.handle_refresh(FakeRequest()))
refresh_payload = body_json(refresh_response)
check("handle_refresh clears the schema memo", memo_cleared == [True])
check("handle_refresh reports the model count",
      refresh_response.status == 200
      and refresh_payload == {"models": len(SAMPLE_MODELS), "provider": "atlascloud"},
      str(refresh_payload))
check("handle_refresh forces a catalogue refresh",
      list_models_calls[-1].get("force_refresh") is True, str(list_models_calls[-1]))

am.clear_schema_memo = original_clear_schema_memo


def failing_clear_schema_memo():
    raise RuntimeError("memo lock stuck")


am.clear_schema_memo = failing_clear_schema_memo
refresh_error_response = run(routes.handle_refresh(FakeRequest()))
refresh_error_payload = body_json(refresh_error_response)
check("handle_refresh returns HTTP 502 when clear_schema_memo raises",
      refresh_error_response.status == 502)
check("handle_refresh 502 payload carries the error message",
      refresh_error_payload.get("error") == "memo lock stuck", str(refresh_error_payload))

am.clear_schema_memo = original_clear_schema_memo

# --- register_routes: a second registration is a no-op ----------------------
# The README recommends a hot-reload extension that re-imports custom node
# modules without restarting ComfyUI; register_routes() then runs a second
# time (from a fresh module namespace) against the *same* PromptServer.
# instance.routes object it registered against the first time. Simulate
# aiohttp's real behaviour (raising on a duplicate route registration) so the
# guard is actually proven, not just assumed.
class FakeRouteTable:
    def __init__(self):
        self.registered = []

    def _decorator_for(self, method, path):
        def decorator(handler):
            if (method, path) in self.registered:
                raise RuntimeError(
                    f"Added route will never be executed, method {method} path '{path}'"
                )
            self.registered.append((method, path))
            return handler
        return decorator

    def get(self, path):
        return self._decorator_for("GET", path)

    def post(self, path):
        return self._decorator_for("POST", path)


class FakePromptServerInstance:
    def __init__(self):
        self.routes = FakeRouteTable()


class FakePromptServer:
    instance = FakePromptServerInstance()


original_prompt_server = routes.PromptServer
routes.PromptServer = FakePromptServer

routes.register_routes()
check("the first register_routes call registers all three routes",
      len(FakePromptServer.instance.routes.registered) == 3,
      str(FakePromptServer.instance.routes.registered))

try:
    routes.register_routes()
    second_call_raised = False
except RuntimeError:
    second_call_raised = True
check("a second register_routes call does not raise (aiohttp would reject the duplicate route)",
      not second_call_raised)
check("a second register_routes call registers nothing further",
      len(FakePromptServer.instance.routes.registered) == 3,
      str(FakePromptServer.instance.routes.registered))
check("the guard is stored on the PromptServer instance, not the module "
      "(so it survives a hot-reload re-import of this module)",
      getattr(FakePromptServer.instance, routes._REGISTERED_FLAG, False) is True)

# A second, independent PromptServer.instance (simulating a second, unrelated
# process/test, not a reload) must be able to register normally -- the guard
# is per-instance, not global.
FakePromptServer.instance = FakePromptServerInstance()
routes.register_routes()
check("a fresh PromptServer instance registers normally (the guard is per-instance)",
      len(FakePromptServer.instance.routes.registered) == 3,
      str(FakePromptServer.instance.routes.registered))

routes.PromptServer = original_prompt_server

# --- import style matches ComfyUI's actual custom-node loader ---------------
# ComfyUI loads a custom node pack with
#   importlib.util.spec_from_file_location(name, .../__init__.py,
#                                           submodule_search_locations=[pack_dir])
# and never appends the pack directory to sys.path. A sibling-module import
# written as `import atlascloud_models` (flat/absolute) only works by accident
# in this test file because line 16 above puts the directory on sys.path --
# under the real loader there is no such entry, so a flat import raises
# ModuleNotFoundError and takes the whole node pack down with it (all 7 node
# classes, not just the Atlas routes). This check reproduces the real loader
# in a subprocess, with a clean sys.path (no PYTHONPATH, cwd outside the
# package directory), so it cannot be fooled by this file's own sys.path
# insertion the way a same-process check could be.
LOADER_REPRODUCTION_SCRIPT = """
import sys
import importlib.util

package_dir = {package_dir!r}
module_name = "comfyui_mdpack_loader_reproduction"

spec = importlib.util.spec_from_file_location(
    module_name,
    package_dir + "/__init__.py",
    submodule_search_locations=[package_dir],
)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

routes_module = sys.modules[module_name + ".atlascloud_routes"]
assert hasattr(routes_module, "handle_models"), "handle_models missing from the loaded routes module"
assert hasattr(routes_module, "handle_schema"), "handle_schema missing from the loaded routes module"
assert hasattr(routes_module, "handle_refresh"), "handle_refresh missing from the loaded routes module"
print("LOADER_REPRODUCTION_OK")
"""

package_directory = os.path.dirname(os.path.abspath(__file__))
clean_environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
loader_result = subprocess.run(
    [sys.executable, "-c", LOADER_REPRODUCTION_SCRIPT.format(package_dir=package_directory)],
    cwd=tempfile.gettempdir(),  # deliberately outside the package directory
    capture_output=True,
    text=True,
)
check(
    "package loads the way ComfyUI's custom-node loader loads it "
    "(spec_from_file_location + submodule_search_locations, no sys.path append)",
    loader_result.returncode == 0 and "LOADER_REPRODUCTION_OK" in loader_result.stdout,
    f"returncode={loader_result.returncode} stdout={loader_result.stdout!r} "
    f"stderr={loader_result.stderr[-2000:]!r}",
)

# --- a genuine failure inside atlascloud_models must not be masked ----------
# The naive dual-import ("try relative, except ImportError: try flat") looks
# right but has a trap: if atlascloud_models itself fails for a real reason
# while being imported relatively (e.g. it depends on `requests` and that is
# not installed), ModuleNotFoundError is an ImportError, so the except clause
# catches it and retries the flat import -- which, under the real ComfyUI
# loader (no sys.path entry for the pack directory), also fails, but with
# "No module named 'atlascloud_models'" instead of the true "No module named
# 'requests'". That masks the real cause behind a misleading one.
#
# This is reproduced with a synthetic package (a bare types.ModuleType whose
# __path__ points at a temp directory containing a stub atlascloud_models.py
# that raises ModuleNotFoundError("No module named 'requests'"), plus the real
# atlascloud_routes.py) so the failure is guaranteed to originate from exactly
# the line under test, not from some unrelated `requests` import elsewhere in
# the node pack. Run in a subprocess so it cannot see the outer sys.modules
# cache built up by the rest of this file.
missing_dependency_directory = tempfile.mkdtemp(prefix="mdpack_missing_dependency_")
with open(
    os.path.join(missing_dependency_directory, "atlascloud_models.py"), "w", encoding="utf-8"
) as stub_atlascloud_models_file:
    stub_atlascloud_models_file.write(
        "raise ModuleNotFoundError(\"No module named 'requests'\")\n"
    )
shutil.copy(
    os.path.join(package_directory, "atlascloud_routes.py"),
    os.path.join(missing_dependency_directory, "atlascloud_routes.py"),
)

MISSING_DEPENDENCY_SCRIPT = """
import sys
import types
import importlib

package_dir = "__PACKAGE_DIR__"
package_name = "mdpack_missing_dependency_check"

synthetic_package = types.ModuleType(package_name)
synthetic_package.__path__ = [package_dir]
sys.modules[package_name] = synthetic_package

try:
    importlib.import_module(package_name + ".atlascloud_routes")
    print("UNEXPECTED_SUCCESS")
except Exception as error:
    print("IMPORT_FAILED: " + type(error).__name__ + ": " + str(error))
"""

missing_dependency_result = subprocess.run(
    [sys.executable, "-c", MISSING_DEPENDENCY_SCRIPT.replace(
        "__PACKAGE_DIR__", missing_dependency_directory
    )],
    cwd=tempfile.gettempdir(),  # deliberately outside both package directories
    capture_output=True,
    text=True,
)
missing_dependency_output = missing_dependency_result.stdout
check(
    "a genuine failure inside atlascloud_models (e.g. a missing dependency) surfaces "
    "its real cause instead of a misleading 'atlascloud_models not found'",
    missing_dependency_result.returncode == 0
    and "No module named 'requests'" in missing_dependency_output
    and "atlascloud_models" not in missing_dependency_output,
    f"stdout={missing_dependency_output!r} stderr={missing_dependency_result.stderr[-2000:]!r}",
)

# --- module import without ComfyUI present -----------------------------------
# atlascloud_routes was already imported above (with PromptServer unavailable,
# since this environment has no ComfyUI `server` module on the path), which
# proves import does not raise. register_routes() must have been a no-op --
# there is no PromptServer.instance to have registered anything against.
check("module imports without ComfyUI present", "atlascloud_routes" in sys.modules)
check("PromptServer is unavailable outside ComfyUI, so no instance to register against",
      routes.PromptServer is None)

# --- provider selection and key transport -------------------------------------
# WaveSpeed's catalogue needs a key. It arrives in a header, never in the query
# string, so it does not end up in the aiohttp access log.
am.list_models = fake_list_models
am.schema_for_model = fake_schema_for_model
list_models_calls.clear()

provider_response = run(routes.handle_models(FakeRequest(
    {"provider": "wavespeed", "output": "Video"},
    {routes.API_KEY_HEADER: "ws-secret"},
)))
provider_payload = body_json(provider_response)
check("handle_models forwards the requested provider",
      list_models_calls[-1]["provider"] == "wavespeed", str(list_models_calls[-1]))
check("handle_models reads the API key from the header",
      list_models_calls[-1]["api_key"] == "ws-secret", str(list_models_calls[-1]))
check("the response names the provider it answered for",
      provider_payload["provider"] == "wavespeed"
      and provider_payload["providers"] == am.PROVIDER_IDS, str(provider_payload))

check("a request without the header sends no key",
      run(routes.handle_models(FakeRequest({"provider": "wavespeed"}))) is not None
      and list_models_calls[-1]["api_key"] == "", str(list_models_calls[-1]))

unknown_provider_response = run(routes.handle_models(FakeRequest({"provider": "wavespeeed"})))
check("an unknown provider is a 400, not a silent fallback to Atlas Cloud",
      unknown_provider_response.status == 400, str(body_json(unknown_provider_response)))

schema_calls.clear()
schema_provider_response = run(routes.handle_schema(FakeRequest(
    {"model": "wavespeed-ai/flux-2-pro/text-to-image", "provider": "wavespeed"},
    {routes.API_KEY_HEADER: "ws-secret"},
)))
check("handle_schema answers for the requested provider",
      schema_provider_response.status == 200
      and schema_calls[-1] == "wavespeed-ai/flux-2-pro/text-to-image", str(schema_calls))
check("handle_schema rejects an unknown provider with 400",
      run(routes.handle_schema(FakeRequest({"model": "x/y", "provider": "nope"}))).status == 400)

list_models_calls.clear()
refresh_provider_response = run(routes.handle_refresh(FakeRequest(
    {"provider": "wavespeed"}, {routes.API_KEY_HEADER: "ws-secret"},
)))
check("handle_refresh refreshes the provider it was asked for, with the key",
      body_json(refresh_provider_response)["provider"] == "wavespeed"
      and list_models_calls[-1]["provider"] == "wavespeed"
      and list_models_calls[-1]["api_key"] == "ws-secret", str(list_models_calls[-1]))

print()
if FAILURES:
    print(f"{len(FAILURES)} TEST(S) FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("ALL TESTS PASSED")
