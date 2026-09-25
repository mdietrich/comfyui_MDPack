"""HTTP routes that hand a provider's catalogue to the node's web extension.

ComfyUI's frontend cannot call the provider APIs directly (no CORS headers
there), so the catalogue and the per-model schemas travel through the ComfyUI
server, which also means the browser benefits from the same disk cache the
nodes use.

The provider is chosen per request with ?provider=; WaveSpeed additionally
needs an API key, which the browser sends in the X-MDPack-Api-Key header
rather than the query string, so it stays out of the aiohttp access log.
"""

from aiohttp import web

# `from . import ...` only fails when there is no parent package to resolve --
# exactly what `__package__` already tells us -- so branching on it (rather than
# catching the ImportError) means a genuine failure inside atlascloud_models
# itself (e.g. a missing dependency) propagates with its real message instead
# of being masked by a second, misleading "no module named atlascloud_models"
# from a doomed flat-import retry.
if __package__:
    from . import atlascloud_models as models
else:  # imported as a loose module (tests, tooling), not as a package submodule
    import atlascloud_models as models

try:
    from server import PromptServer
except ImportError:  # allows importing the module outside of ComfyUI
    PromptServer = None


API_KEY_HEADER = "X-MDPack-Api-Key"


def _request_provider(request):
    """The provider this request asks for, or an error response for a bad one."""
    try:
        provider, _ = models.provider_config(request.query.get("provider", "") or "")
    except ValueError as error:
        return None, web.json_response({"error": str(error), "models": []}, status=400)
    return provider, None


def _request_api_key(request):
    return request.headers.get(API_KEY_HEADER, "") or ""


async def handle_models(request):
    output_kind = request.query.get("output", "") or ""
    input_kind = request.query.get("input", "") or ""
    force_refresh = request.query.get("refresh", "") in ("1", "true", "yes")
    provider, error_response = _request_provider(request)
    if error_response is not None:
        return error_response
    try:
        entries = models.list_models(
            output_kind=output_kind, input_kind=input_kind, force_refresh=force_refresh,
            provider=provider, api_key=_request_api_key(request),
        )
    except Exception as error:
        return web.json_response({"error": str(error), "models": []}, status=502)
    return web.json_response({
        "models": entries,
        "provider": provider,
        "providers": models.PROVIDER_IDS,
        "output_kinds": models.OUTPUT_KINDS,
        "input_kinds": models.INPUT_KINDS,
    })


async def handle_schema(request):
    model_identifier = request.query.get("model", "") or ""
    if not model_identifier:
        return web.json_response({"error": "model parameter missing"}, status=400)
    provider, error_response = _request_provider(request)
    if error_response is not None:
        return error_response
    try:
        return web.json_response(models.schema_for_model(
            model_identifier, provider=provider, api_key=_request_api_key(request)
        ))
    except Exception as error:
        return web.json_response({"error": str(error)}, status=502)


async def handle_refresh(request):
    provider, error_response = _request_provider(request)
    if error_response is not None:
        return error_response
    try:
        models.clear_schema_memo()
        entries = models.list_models(
            force_refresh=True, provider=provider, api_key=_request_api_key(request)
        )
    except Exception as error:
        return web.json_response({"error": str(error)}, status=502)
    return web.json_response({"models": len(entries), "provider": provider})


_REGISTERED_FLAG = "_mdpack_atlascloud_routes_registered"


def register_routes():
    """Register the three Atlas Cloud routes, exactly once per PromptServer.

    The README recommends a hot-reload extension (e.g. LG_HotReload), which
    re-imports custom node modules -- including this one -- without
    restarting ComfyUI. A fresh import gives register_routes() a brand new
    module namespace, so a module-level "already ran" flag would not survive
    the reload; the guard has to live on PromptServer.instance itself, the
    one object that *does* persist across a reload, or aiohttp raises on the
    second registration of the same path (see the final review, Finding 9).
    """
    if PromptServer is None or not hasattr(PromptServer, "instance"):
        return
    instance = PromptServer.instance
    if getattr(instance, _REGISTERED_FLAG, False):
        return
    routes = instance.routes
    routes.get("/mdpack/atlas/models")(handle_models)
    routes.get("/mdpack/atlas/schema")(handle_schema)
    routes.post("/mdpack/atlas/refresh")(handle_refresh)
    setattr(instance, _REGISTERED_FLAG, True)


register_routes()
