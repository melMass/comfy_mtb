# NOTE: this file must be used in all test_* files to setup the test env.

import importlib
import logging
import sys
from pathlib import Path


def setup_test():
    mod_path = Path(__file__).parent.parent

    # add custom_nodes to path
    sys.path.append(mod_path.parent.as_posix())
    print(f"Appended: {mod_path.parent.as_posix()}")

    # add comfy root to path
    sys.path.append(mod_path.parent.parent.as_posix())
    print(f"Appended: {mod_path.parent.parent.as_posix()}")

    # import mtb
    module = importlib.import_module(mod_path.name)

    # add the module to globals
    globals()[mod_path.name] = module

    # set the logging level for third-party libraries
    logging.getLogger("xformers").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("h5py._conv").setLevel(logging.ERROR)
    logging.getLogger("numexpr.utils").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
    logging.getLogger("torch.utils.tensorboard").setLevel(logging.ERROR)
    logging.getLogger("torchvision.transforms.functional_tensor").setLevel(
        logging.ERROR
    )
    logging.getLogger("basicsr.metrics.niqe").setLevel(logging.ERROR)

