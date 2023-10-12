#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: __init__.py
# Project: comfy_mtb
# Author: Mel Massadian
# Copyright (c) 2023 Mel Massadian
#
###
import os

# todo: don't override this if the user has that setup already
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import ast
import contextlib
import importlib
import json
import logging
import os
import shutil
import traceback
from importlib import reload

from aiohttp import web
from server import PromptServer

import nodes

from .endpoint import endlog
from .log import blue_text, cyan_text, get_label, get_summary, log
from .utils import comfy_dir, here

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CLASS_MAPPINGS_DEBUG = {}
WEB_DIRECTORY = "./web"

__version__ = "0.2.0"


def extract_nodes_from_source(filename):
    source_code = ""

    with open(filename, "r", encoding="utf8") as file:
        source_code = file.read()

    nodes = []

    try:
        parsed = ast.parse(source_code)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id == "__nodes__":
                    value = ast.get_source_segment(source_code, node.value)
                    node_value = ast.parse(value).body[0].value
                    if isinstance(node_value, (ast.List, ast.Tuple)):
                        nodes.extend(
                            element.id
                            for element in node_value.elts
                            if isinstance(element, ast.Name)
                        )
                    break
    except SyntaxError:
        log.error("Failed to parse")
    return nodes


def load_nodes():
    errors = []
    nodes = []
    nodes_failed = []

    for filename in (here / "nodes").iterdir():
        if filename.suffix == ".py":
            module_name = filename.stem

            try:
                module = importlib.import_module(
                    f".nodes.{module_name}", package=__package__
                )
                _nodes = getattr(module, "__nodes__")
                nodes.extend(_nodes)
                log.debug(f"Imported {module_name} nodes")

            except AttributeError:
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
            f"Some nodes failed to load:\n\t"
            + "\n\t".join(errors)
            + "\n\n"
            + "Check that you properly installed the dependencies.\n"
            + "If you think this is a bug, please report it on the github page (https://github.com/melMass/comfy_mtb/issues)"
        )

    return (nodes, nodes_failed)


# - REGISTER WEB EXTENSIONS
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
            f"Failed to remove web mtb directory: {e}\nPlease manually remove it from disk ({web_mtb}) and restart the server."
        )


# - REGISTER NODES
nodes, failed = load_nodes()
for node_class in nodes:
    class_name = node_class.__name__
    node_label = f"{get_label(class_name)} (mtb)"
    NODE_CLASS_MAPPINGS[node_label] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[class_name] = node_label
    NODE_CLASS_MAPPINGS_DEBUG[node_label] = node_class.__doc__
    # TODO: I removed this, I find it more convenient to write without spaces, but it breaks every of my workflows
    # TODO (cont): and until I find a way to automate the conversion, I'll leave it like this

    if os.environ.get("MTB_EXPORT"):
        with open(here / "node_list.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        k: NODE_CLASS_MAPPINGS_DEBUG[k]
                        for k in sorted(NODE_CLASS_MAPPINGS_DEBUG.keys())
                    },
                    indent=4,
                )
            )

log.debug(
    f"Loaded the following nodes:\n\t"
    + "\n\t".join(
        f"{cyan_text(k)}: {blue_text(get_summary(doc)) if doc else '-'}"
        for k, doc in NODE_CLASS_MAPPINGS_DEBUG.items()
    )
)

log.info(f"loaded {cyan_text(len(nodes))} nodes successfuly")
if failed:
    with contextlib.suppress(Exception):
        base_url, port = utils.get_server_info()
        log.info(
            f"Some nodes ({len(failed)}) could not be loaded. This can be ignored, but go to http://{base_url}:{port}/mtb if you want more information."
        )


# - ENDPOINT


if hasattr(PromptServer, "instance"):
    restore_deps = ["basicsr"]
    onnx_deps = ["onnxruntime"]
    swap_deps = ["insightface"] + onnx_deps
    node_dependency_mapping = {
        "QrCode": ["qrcode"],
        "DeepBump": onnx_deps,
        "FaceSwap": swap_deps,
        "LoadFaceSwapModel": swap_deps,
        "LoadFaceAnalysisModel": restore_deps,
    }

    PromptServer.instance.app.router.add_static(
        "/mtb-assets/", path=(here / "html").as_posix()
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

    @PromptServer.instance.routes.post("/mtb/debug")
    async def set_debug(request):
        json_data = await request.json()
        enabled = json_data.get("enabled")
        if enabled:
            os.environ["MTB_DEBUG"] = "true"
            log.setLevel(logging.DEBUG)
            log.debug("Debug mode set from API (/mtb/debug POST route)")

        elif "MTB_DEBUG" in os.environ:
            # del os.environ["MTB_DEBUG"]
            os.environ.pop("MTB_DEBUG")
            log.setLevel(logging.INFO)

        return web.json_response(
            {"message": f"Debug mode {'set' if enabled else 'unset'}"}
        )

    @PromptServer.instance.routes.get("/mtb")
    async def get_home(request):
        from . import endpoint

        reload(endpoint)
        # Check if the request prefers HTML content
        if "text/html" in request.headers.get("Accept", ""):
            # # Return an HTML page
            html_response = """
            <div class="flex-container menu">
                <a href="/mtb/manage">manage</a>
                <a href="/mtb/debug">debug</a>
                <a href="/mtb/status">status</a>
            </div>            
            """
            return web.Response(
                text=endpoint.render_base_template("MTB", html_response),
                content_type="text/html",
            )

        # Return JSON for other requests
        return web.json_response({"message": "Welcome to MTB!"})

    @PromptServer.instance.routes.get("/mtb/debug")
    async def get_debug(request):
        from . import endpoint

        reload(endpoint)
        enabled = "MTB_DEBUG" in os.environ
        # Check if the request prefers HTML content
        if "text/html" in request.headers.get("Accept", ""):
            # # Return an HTML page
            html_response = f"""
                <h1>MTB Debug Status: {'Enabled' if enabled else 'Disabled'}</h1>
            """
            return web.Response(
                text=endpoint.render_base_template("Debug", html_response),
                content_type="text/html",
            )

        # Return JSON for other requests
        return web.json_response({"enabled": enabled})

    @PromptServer.instance.routes.get("/mtb/actions")
    async def no_route(request):
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
    async def do_action(request):
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
