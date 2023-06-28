import traceback
from .log import log, blue_text, get_summary, get_label
from .utils import here
import importlib
import os

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS_DEBUG = {}


def load_nodes():
    errors = []
    nodes = []
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
                errors.append(f"Failed to import {module_name} because {error_message}")

    if errors:
        log.error(
            f"Some nodes failed to load:\n\t"
            + "\n\t".join(errors)
            + "\n\n"
            + "Check that you properly installed the dependencies.\n"
            + "If you think this is a bug, please report it on the github page (https://github.com/melMass/comfy_mtb/issues)"
        )

    return nodes


# - REGISTER WEB EXTENSIONS
web_extensions_root = utils.comfy_dir / "web" / "extensions"
web_mtb = web_extensions_root / "mtb"

if web_mtb.exists():
    log.debug(f"Web extensions folder found at {web_mtb}")
elif web_extensions_root.exists():
    os.symlink((here / "web"), web_mtb.as_posix())
else:
    log.error(
        f"Comfy root probably not found automatically, please copy the folder {web_mtb} manually in the web/extensions folder of ComfyUI"
    )

# - REGISTER NODES
nodes = load_nodes()
for node_class in nodes:
    class_name = node_class.__name__
    class_name = node_class.__name__
    node_name = f"{get_label(class_name)} (mtb)"
    NODE_CLASS_MAPPINGS[node_name] = node_class
    NODE_CLASS_MAPPINGS_DEBUG[node_name] = node_class.__doc__


log.debug(
    f"Loaded the following nodes:\n\t"
    + "\n\t".join(
        f"{k}: {blue_text(get_summary(doc)) if doc else '-'}"
        for k, doc in NODE_CLASS_MAPPINGS_DEBUG.items()
    )
)
