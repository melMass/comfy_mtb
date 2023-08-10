from ..utils import here
from ..log import log
import folder_paths
from pathlib import Path
import shutil
import csv


class SmartStep:
    """Utils to control the steps start/stop of the KAdvancedSampler in percentage"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "step": (
                    "INT",
                    {"default": 20, "min": 1, "max": 10000, "step": 1},
                ),
                "start_percent": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1},
                ),
                "end_percent": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("step", "start", "end")
    FUNCTION = "do_step"
    CATEGORY = "mtb/conditioning"

    def do_step(self, step, start_percent, end_percent):
        start = int(step * start_percent / 100)
        end = int(step * end_percent / 100)

        return (step, start, end)


def install_default_styles(force=False):
    styles_dir = Path(folder_paths.base_path) / "styles"
    styles_dir.mkdir(parents=True, exist_ok=True)
    default_style = here / "styles.csv"
    dest_style = styles_dir / "default.csv"

    if force or not dest_style.exists():
        log.debug(f"Copying default style to {dest_style}")
        shutil.copy2(default_style.as_posix(), dest_style.as_posix())

    return dest_style


class StylesLoader:
    """Load csv files and populate a dropdown from the rows (à la A111)"""

    options = {}

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.options:
            input_dir = Path(folder_paths.base_path) / "styles"
            if not input_dir.exists():
                install_default_styles()

            if not (files := [f for f in input_dir.iterdir() if f.suffix == ".csv"]):
                log.warn(
                    "No styles found in the styles folder, place at least one csv file in the styles folder at the root of ComfyUI (for instance ComfyUI/styles/mystyle.csv)"
                )

            for file in files:
                with open(file, "r", encoding="utf8") as f:
                    parsed = csv.reader(f)
                    for row in parsed:
                        log.debug(f"Adding style {row[0]}")
                        cls.options[row[0]] = (row[1], row[2])

        else:
            log.debug(f"Using cached styles (count: {len(cls.options)})")

        return {
            "required": {
                "style_name": (list(cls.options.keys()),),
            }
        }

    CATEGORY = "mtb/conditioning"

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "load_style"

    def load_style(self, style_name):
        return (self.options[style_name][0], self.options[style_name][1])


__nodes__ = [SmartStep, StylesLoader]
