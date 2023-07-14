import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import traceback
from .log import log, blue_text, cyan_text, get_summary, get_label
from .utils import here
import importlib
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
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
    try:
        os.symlink((here / "web"), web_mtb.as_posix())
    except Exception:  # OSError
        log.error(
            f"Failed to create symlink to {web_mtb}. Please copy the folder manually."
        )
else:
    log.error(
        f"Comfy root probably not found automatically, please copy the folder {web_mtb} manually in the web/extensions folder of ComfyUI"
    )

# - REGISTER NODES
nodes = load_nodes()
for node_class in nodes:
    class_name = node_class.__name__
    node_label = f"{get_label(class_name)} (mtb)"
    NODE_CLASS_MAPPINGS[node_label] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[class_name] = node_label
    NODE_CLASS_MAPPINGS_DEBUG[node_label] = node_class.__doc__
    # TODO: I removed this, I find it more convenient to write without spaces, but it breaks every of my workflows
    # TODO (cont): and until I find a way to automate the conversion, I'll leave it like this

log.info(
    f"Loaded the following nodes:\n\t"
    + "\n\t".join(
        f"{cyan_text(k)}: {blue_text(get_summary(doc)) if doc else '-'}"
        for k, doc in NODE_CLASS_MAPPINGS_DEBUG.items()
    )
)
