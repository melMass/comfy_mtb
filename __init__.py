import traceback
from .log import log
from .utils import here
from pathlib import Path
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS_DEBUG = {}


def load_nodes():
    errors = []
    nodes = []
    for filename in (here  / "nodes").iterdir():
        if filename.suffix == '.py':
            module_name = filename.stem
            module_path = filename.resolve().as_posix()
            
            try:
                module = importlib.import_module(f".nodes.{module_name}",package=__package__)
                _nodes = getattr(module, '__nodes__')
                
                nodes.extend(_nodes)
                # Use the `nodes` variable here as needed
                log.debug(f"Imported __nodes__ from {module_name}")
            
            except Exception:
                error_message = traceback.format_exc().splitlines()[-1]
                errors.append(f"Failed to import {module_name}. {error_message}")
                # log.error(f"Failed to import {module_name}. {error_message}")

    if errors:
        log.error(f"Some nodes failed to load:\n\t" + "\n\t".join(errors) + "\n\n" + "Check that you properly installed the dependencies.\n" + "If you think this is a bug, please report it on the github page (https://github.com/melMass/comfy_mtb/issues)")

    return nodes
nodes = load_nodes()

for node_class in nodes:
    class_name = node_class.__name__
    class_name = node_class.__name__
    
    NODE_CLASS_MAPPINGS[class_name] = node_class
    NODE_CLASS_MAPPINGS_DEBUG[class_name] = node_class.__doc__
    
    
def get_summary(docstring):
    return docstring.strip().split('\n\n', 1)[0]

log.debug(f"Loaded the following nodes:\n\t" + "\n\t".join(f"{k}: {get_summary(doc) if doc else '-'}" for k,doc in NODE_CLASS_MAPPINGS_DEBUG.items()))
