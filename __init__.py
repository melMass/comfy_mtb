#!/usr/bin/env python3
###
# File: __init__.py
# Project: comfy_mtb
# Author: Mel Massadian
# Copyright (c) 2023-2025 Mel Massadian
#
###

__version__ = "0.5.4"

import os

from aiohttp.web_request import Request

# TODO: don't override this if the user has that setup already
if not os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if not os.environ.get("TF_GPU_ALLOCATOR"):
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import ast
import contextlib
import importlib
import json
import logging
import shutil
import traceback
from importlib import reload
from pathlib import Path

from aiohttp import web

IN_COMFY = False

try:
    from server import PromptServer

    IN_COMFY = True
except ModuleNotFoundError:
    IN_COMFY = False


from .endpoint import endlog
from .install import get_node_dependencies
from .log import blue_text, cyan_text, get_label, get_summary, log
from .utils import comfy_dir, here

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
NODE_CLASS_MAPPINGS_DEBUG: dict[str, str | None] = {}
WEB_DIRECTORY = "./web"


def extract_nodes_from_source(filename: Path):
    source_code = ""
    source_code = filename.read_text(encoding="utf-8")
    nodes: list[str] = []

    try:
        parsed = ast.parse(source_code)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == "__nodes__":
                    value = ast.get_source_segment(source_code, node.value)
                    if value:
                        node_value = ast.parse(value).body[0].value
                        if isinstance(node_value, ast.List | ast.Tuple):
                            nodes.extend(
                                str(element.id)
                                for element in node_value.elts
                                if isinstance(element, ast.Name)
                            )
                        break
    except SyntaxError:
        log.error("Failed to parse")
    return nodes


def load_nodes():
    errors: list[str] = []
    nodes: list[type] = []
    nodes_failed: list[str] = []

    for filename in (here / "nodes").iterdir():
        if filename.suffix == ".py":
            module_name = filename.stem

            try:
                module = importlib.import_module(
                    f".nodes.{module_name}", package=__package__
                )
                _nodes = getattr(module, "__nodes__", [])
                nodes.extend(_nodes)
                log.debug(f"Imported {module_name} nodes")

            except AttributeError:
                log.debug(f"Skipping wip module {module_name}")
                pass  # wip nodes
            except Exception:
                error_message = traceback.format_exc().splitlines()[-1]

                errors.append(
                    f"Failed to import module {module_name} because {error_message}"
                )
                # Read __nodes__ variable from the source file
                nodes_failed.extend(extract_nodes_from_source(filename))

    if errors:
        log.debug(
            "Some nodes failed to load:\n\t"
            + "\n\t".join(errors)
            + "\n\n"
            + "Check that you properly installed the dependencies.\n"
            + "If you think this is a bug, please report it on the github page (https://github.com/melMass/comfy_mtb/issues)"
        )

    return (nodes, nodes_failed)


# - REGISTER WEB EXTENSIONS
def uninstall_old_web_extensions():
    web_extensions_root = comfy_dir / "web" / "extensions"
    web_mtb = web_extensions_root / "mtb"

    if web_mtb.exists() and hasattr(nodes, "EXTENSION_WEB_DIRS"):
        try:
            if web_mtb.is_symlink():
                web_mtb.unlink()
            else:
                shutil.rmtree(web_mtb)
        except Exception as e:
            log.warning(
                f"""Failed to remove web mtb directory: {e}
Please manually remove it from disk ({web_mtb}) and restart the server."""
            )


# uninstall_old_web_extensions()


# - GATHER WIKI PAGES
def wiki_to_classname(s: str):
    wiki_name = s.replace("nodes-", "", 1)
    return "MTB_" + "".join(
        [part.capitalize() for part in wiki_name.split("-")]
    )


def classname_to_wiki(s: str):
    classname = s.replace("MTB_", "")
    parts: list[str] = []
    start = 0
    for i in range(1, len(classname)):
        if classname[i].isupper():
            parts.append(classname[start:i].lower())
            start = i
    parts.append(classname[start:].lower())
    return "nodes-" + "-".join(parts)


wiki = here / "wiki"
node_docs = {}
if wiki.exists() and wiki.is_dir():
    node_docs = {
        wiki_to_classname(x.stem): x.read_text(encoding="utf-8")
        for x in (wiki / "nodes").glob("*.md")
    }


# - REGISTER NODES
MTB_EXPORT = os.environ.get("MTB_EXPORT")

nodes, failed = load_nodes()
for node_class in nodes:
    class_name: str = node_class.__name__
    linked_doc = node_docs.get(class_name)

    if not hasattr(node_class, "DESCRIPTION"):
        if linked_doc:
            log.debug(f"Found linked doc for {class_name}, using it")
            node_class.DESCRIPTION = linked_doc
        elif node_class.__doc__:
            log.debug(f"Using __doc__ as description for {class_name}")
            node_class.DESCRIPTION = node_class.__doc__
            if MTB_EXPORT:
                wiki_name = classname_to_wiki(class_name)
                _ = (wiki / "nodes" / (wiki_name + ".md")).write_text(
                    node_class.__doc__, encoding="utf-8"
                )

        else:
            log.debug(
                f"None of the methods could retrieve documentation for {class_name}"
            )

    node_label = f"{get_label(class_name)} (mtb)"
    NODE_CLASS_MAPPINGS[node_label] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[class_name] = node_label
    NODE_CLASS_MAPPINGS_DEBUG[node_label] = node_class.__doc__

    # TODO: I removed this, I find it more convenient to write without spaces
    # but it breaks every of my workflows
    # TODO (cont): and until I find a way to automate the conversion
    # I'll leave it like this

    if os.environ.get("MTB_EXPORT"):
        with open(here / "node_list.json", "w") as f:
            _ = f.write(
                json.dumps(
                    {
                        k: NODE_CLASS_MAPPINGS_DEBUG[k]
                        for k in sorted(NODE_CLASS_MAPPINGS_DEBUG.keys())
                    },
                    indent=4,
                )
            )

log.debug(
    "Loaded the following nodes:\n\t"
    + "\n\t".join(
        f"{cyan_text(k)}: {blue_text(get_summary(doc)) if doc else '-'}"
        for k, doc in NODE_CLASS_MAPPINGS_DEBUG.items()
    )
)

log.info(f"loaded {cyan_text(str(len(nodes)))} nodes successfuly")

if failed:
    with contextlib.suppress(Exception):
        base_url, port = utils.get_server_info()
        log.info(
            f"Some nodes ({len(failed)}) could not be loaded. This can be ignored, but go to http://{base_url}:{port}/mtb if you want more information."
        )
        log.debug(failed)


# - ENDPOINT


if IN_COMFY and hasattr(PromptServer, "instance"):
    img_cache = None
    prompt_cache = None

    with contextlib.suppress(ImportError):
        from cachetools import TTLCache

        img_cache = TTLCache(maxsize=100, ttl=5)  # 1 min TTL
        prompt_cache = TTLCache(maxsize=100, ttl=5)  # 1 min TTL

    node_dependency_mapping = get_node_dependencies()

    PromptServer.instance.app.router.add_static(
        "/mtb-assets/", path=(here / "html").as_posix()
    )

    # NOTE: we add an extra static path to avoid comfy mechanism
    # that loads every script in web.
    PromptServer.instance.app.add_routes(
        [web.static("/mtb_async", (here / "web_async").as_posix())]
    )

    @PromptServer.instance.routes.get("/mtb/manage")
    async def manage(request):
        from . import endpoint

        reload(endpoint)

        endlog.debug("Initializing Manager")
        if "text/html" in request.headers.get("Accept", ""):
            csv_editor = endpoint.csv_editor()

            tabview = endpoint.render_tab_view(Styles=csv_editor)
            return web.Response(
                text=endpoint.render_base_template("MTB", tabview),
                content_type="text/html",
            )

        return web.json_response(
            {
                "message": "manage only has a POST api for now",
            }
        )

    @PromptServer.instance.routes.get("/mtb/status")
    async def get_full_library(request):
        from . import endpoint

        reload(endpoint)

        endlog.debug("Getting node registration status")
        # Check if the request prefers HTML content
        if "text/html" in request.headers.get("Accept", ""):
            # # Return an HTML page
            html_response = endpoint.render_table(
                NODE_CLASS_MAPPINGS_DEBUG, title="Registered"
            )
            html_response += endpoint.render_table(
                {
                    k: {"dependencies": node_dependency_mapping.get(k)}
                    if node_dependency_mapping.get(k)
                    else "-"
                    for k in failed
                },
                title="Failed to load",
            )

            return web.Response(
                text=endpoint.render_base_template("MTB", html_response),
                content_type="text/html",
            )

        return web.json_response(
            {
                "registered": NODE_CLASS_MAPPINGS_DEBUG,
                "failed": failed,
            }
        )

    @PromptServer.instance.routes.post("/mtb/server-info")
    async def set_server_info(request: Request):
        json_data: dict[str, bool] = await request.json()
        enabled = json_data.get("debug")
        if enabled:
            os.environ["MTB_DEBUG"] = "true"
            log.setLevel(logging.DEBUG)
            log.debug("Debug mode set from API (/mtb/debug POST route)")

        elif "MTB_DEBUG" in os.environ:
            # del os.environ["MTB_DEBUG"]
            _ = os.environ.pop("MTB_DEBUG")
            log.setLevel(logging.INFO)

        return web.json_response(
            {"message": f"Debug mode {'set' if enabled else 'unset'}"}
        )

    @PromptServer.instance.routes.get("/mtb")
    async def get_home(request: Request):
        from . import endpoint

        _ = reload(endpoint)
        # Check if the request prefers HTML content
        if "text/html" in request.headers.get("Accept", ""):
            # # Return an HTML page
            html_response = """
            <div class="flex-container menu">
                <a href="/mtb/manage">manage</a>
                <a href="/mtb/server-info">Server Info</a>
                <a href="/mtb/status">status</a>
            </div>
            """
            return web.Response(
                text=endpoint.render_base_template("MTB", html_response),
                content_type="text/html",
            )

        # Return JSON for other requests
        return web.json_response({"message": "Welcome to MTB!"})

    import asyncio
    import os
    from io import BytesIO

    from aiohttp import web
    from PIL import Image

    def get_cached_image(file_path: str, preview_params=None, channel=None):
        cache_key = (file_path, preview_params, channel)
        if img_cache and (cache_key in img_cache):
            return img_cache[cache_key]

        with Image.open(file_path) as img:
            info = img.info
            if preview_params:
                img = process_preview(img, preview_params)
            if channel:
                img = process_channel(img, channel)
            if prompt_cache:
                prompt_cache[cache_key] = info
            if img_cache:
                img_cache[cache_key] = img.getvalue()
                return img_cache[cache_key]

            return img.getvalue()

    def process_preview(img: Image.Image, preview_params):
        image_format, quality, width = preview_params
        quality = int(quality)

        if width:
            width = int(width)
            img.thumbnail((width, int(width * img.height / img.width)))

        buffer = BytesIO()
        img.save(
            buffer, format=image_format, quality=quality, metadata=img.info
        )
        buffer.seek(0)
        return buffer

    def process_channel(img: Image.Image, channel: str):
        if channel == "rgb":
            if img.mode == "RGBA":
                r, g, b, _ = img.split()
                img = Image.merge("RGB", (r, g, b))
            else:
                img = img.convert("RGB")
        elif channel == "a":
            if img.mode == "RGBA":
                _, _, _, a = img.split()
            else:
                a = Image.new("L", img.size, 255)
            img = Image.new("RGBA", img.size)
            img.putalpha(a)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        _ = buffer.seek(0)
        return buffer

    async def get_image_response(
        file, filename: str, preview_info=None, channel=None
    ):
        img = await asyncio.to_thread(
            get_cached_image, file, preview_info, channel
        )
        return web.Response(
            body=img,
            content_type="image/webp" if preview_info else "image/png",
            headers={"Content-Disposition": f'filename="{filename}"'},
        )

    # TODO: Embed the metadatas somehow so we can drag and drop
    #       to load workflows in the sidebar
    @PromptServer.instance.routes.get("/mtb/view")
    async def view_image(request: Request):
        import folder_paths

        filename = request.rel_url.query.get("filename")
        if not filename:
            return web.Response(status=404)

        filename, output_dir = folder_paths.annotated_filepath(filename)
        if filename[0] == "/" or ".." in filename:
            return web.Response(status=400)

        if output_dir is None:
            rtype = request.rel_url.query.get("type", "output")
            output_dir = folder_paths.get_directory_by_type(rtype)

        if output_dir is None:
            return web.Response(status=400)

        if "subfolder" in request.rel_url.query:
            full_output_dir = os.path.join(
                output_dir, request.rel_url.query["subfolder"]
            )
            if (
                os.path.commonpath(
                    (os.path.abspath(full_output_dir), output_dir)
                )
                != output_dir
            ):
                return web.Response(status=403)
            output_dir = full_output_dir

        filename = os.path.basename(filename)
        file = os.path.join(output_dir, filename)

        if not os.path.isfile(file):
            return web.Response(status=404)

        ret_workflow = request.rel_url.query.get("workflow")

        if ret_workflow:
            image = Image.open(file)
            prompt = image.info.get("prompt", "")
            workflow = image.info.get("workflow", "")

            if workflow:
                workflow = json.loads(workflow)

            if prompt:
                prompt = json.loads(prompt)

            return web.json_response(
                {
                    "prompt": prompt,
                    "workflow": workflow,
                }
            )

        preview_info = None
        if "preview" in request.rel_url.query:
            preview_params = request.rel_url.query["preview"].split(";")
            image_format = (
                preview_params[0]
                if preview_params[0] in ["webp", "jpeg"]
                else "webp"
            )
            quality = (
                int(preview_params[1])
                if len(preview_params) > 1 and preview_params[1].isdigit()
                else 90
            )
            width = request.rel_url.query.get("width")
            preview_info = (image_format, quality, width)

        channel = request.rel_url.query.get("channel")

        return await get_image_response(file, filename, preview_info, channel)

    @PromptServer.instance.routes.get("/mtb/server-info")
    async def get_debug(request: Request):
        from . import endpoint

        _ = reload(endpoint)
        isdebug = "MTB_DEBUG" in os.environ
        exposed = "MTB_EXPOSE" in os.environ

        def render_property(name: str, val: str):
            return f"""<strong>{name}:</strong>
                <p>
                    {val}
                </p>"""

        # Check if the request prefers HTML content
        if "text/html" in request.headers.get("Accept", ""):
            # # Return an HTML page
            html_response = ""

            html_response += render_property(
                "Debug", "Enabled" if isdebug else "Disabled"
            )

            html_response += render_property("Exposed", str(exposed))

            return web.Response(
                text=endpoint.render_base_template(
                    "Server Info", html_response
                ),
                content_type="text/html",
            )

        # Return JSON for other requests
        return web.json_response({"exposed": exposed, "debug": isdebug})

    @PromptServer.instance.routes.get("/mtb/actions")
    async def no_route(request: Request):
        from . import endpoint

        if "text/html" in request.headers.get("Accept", ""):
            html_response = """
            <h1>Actions has no get for now...</h1>
            """
            return web.Response(
                text=endpoint.render_base_template("Actions", html_response),
                content_type="text/html",
            )
        return web.json_response({"message": "actions has no get for now"})

    @PromptServer.instance.routes.post("/mtb/actions")
    async def do_action(request: Request):
        from . import endpoint

        reload(endpoint)

        return await endpoint.do_action(request)


# - WAS Dictionary
MANIFEST = {
    "name": "MTB Nodes",  # The title that will be displayed on Node Class menu,. and Node Class view
    "version": (0, 1, 0),  # Version of the custom_node or sub module
    "author": "Mel Massadian",  # Author or organization of the custom_node or sub module
    "project": "https://github.com/melMass/comfy_mtb",  # The address that the `name` value will link to on Node Class Views
    "description": "Set of nodes that enhance your animation workflow and provide a range of useful tools including features such as manipulating bounding boxes, perform color corrections, swap faces in images, interpolate frames for smooth animation, export to ProRes format, apply various image operations, work with latent spaces, generate QR codes, and create normal and height maps for textures.",
}
