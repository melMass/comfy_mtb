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

import traceback
from .log import log, blue_text, cyan_text, get_summary, get_label
from .utils import here
from .utils import comfy_dir
import importlib
import os
import ast
import json

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CLASS_MAPPINGS_DEBUG = {}

__version__ = "0.1.4"


def extract_nodes_from_source(filename):
    source_code = ""

    with open(filename, "r") as file:
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
                    if isinstance(node_value, ast.List) or isinstance(
                        node_value, ast.Tuple
                    ):
                        for element in node_value.elts:
                            if isinstance(element, ast.Name):
                                print(element.id)
                                nodes.append(element.id)

                    break
    except SyntaxError:
        log.error("Failed to parse")
        pass  # File couldn't be parsed

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
        log.info(
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

if web_mtb.exists():
    log.debug(f"Web extensions folder found at {web_mtb}")
    if not os.path.islink(web_mtb.as_posix()):
        log.warn(
            f"Web extensions folder at {web_mtb} is not a symlink, if updating please delete it before"
        )


elif web_extensions_root.exists():
    web_tgt = here / "web"
    src = web_tgt.as_posix()
    dst = web_mtb.as_posix()
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(src, dst)
        else:
            os.symlink(web_tgt.as_posix(), web_mtb.as_posix())

    except OSError:
        log.warn(f"Failed to create symlink to {web_mtb}, trying to copy it")
        try:
            import shutil

            shutil.copytree(web_tgt, web_mtb)
            log.info(f"Successfully copied {web_tgt} to {web_mtb}")
        except Exception as e:
            log.warn(
                f"Failed to symlink and copy {web_tgt} to {web_mtb}. Please copy the folder manually."
            )
            log.warn(e)

    except Exception as e:
        log.warn(
            f"Failed to create symlink to {web_mtb}. Please copy the folder manually."
        )
        log.warn(e)
else:
    log.warn(
        f"Comfy root probably not found automatically, please copy the folder {web_mtb} manually in the web/extensions folder of ComfyUI"
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

log.info(
    f"Loaded the following nodes:\n\t"
    + "\n\t".join(
        f"{cyan_text(k)}: {blue_text(get_summary(doc)) if doc else '-'}"
        for k, doc in NODE_CLASS_MAPPINGS_DEBUG.items()
    )
)

# - ENDPOINT
from server import PromptServer
from .log import log
from aiohttp import web
from importlib import reload
import logging
from .endpoint import endlog

if hasattr(PromptServer, "instance"):
    restore_deps = ["basicsr"]
    swap_deps = ["insightface", "onnxruntime"]

    node_dependency_mapping = {
        "FaceSwap": swap_deps,
        "LoadFaceSwapModel": swap_deps,
        "LoadFaceAnalysisModel": restore_deps,
    }

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

        else:
            if "MTB_DEBUG" in os.environ:
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
            html_response = f"""
            <div class="flex-container menu">
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
        enabled = False
        if "MTB_DEBUG" in os.environ:
            enabled = True
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
            html_response = f"""
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
