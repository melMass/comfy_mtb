import csv
import shutil
from pathlib import Path

import folder_paths
import torch

from ..log import log
from ..utils import here

Conditioning = list[tuple[torch.Tensor, dict[str, torch.Tensor]]]


def check_condition(conditioning: Conditioning):
    has_cn = False
    if len(conditioning) > 1:
        log.warn(
            "More than one conditioning was provided. Only the first one will be used."
        )
    first = conditioning[0]
    cond, kwargs = first

    log.debug("Conditioning Shape")
    log.debug(cond.shape)
    log.debug("Conditioning keys")
    log.debug([f"\t{k} - {type(kwargs[k])}" for k in kwargs])
    if "control" in kwargs:
        log.debug("Conditioning contains a controlnet")
        has_cn = True
    if "pooled_output" not in kwargs:
        raise ValueError(
            "Conditioning is not valid. Missing 'pooled_output' key."
        )
    return has_cn


class MTB_InterpolateCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "mtb/conditioning"
    FUNCTION = "execute"

    def execute(
        self, blend: float, **kwargs: Conditioning
    ) -> tuple[Conditioning]:
        blend = max(0.0, min(1.0, blend))

        conditions: list[Conditioning] = list(kwargs.values())
        num_conditions = len(conditions)

        if num_conditions < 2:
            raise ValueError("At least two conditioning inputs are required.")

        segment_length = 1.0 / (num_conditions - 1)

        segment_index = min(int(blend // segment_length), num_conditions - 2)

        local_blend = (
            blend - (segment_index * segment_length)
        ) / segment_length

        cond_from = conditions[segment_index]
        cond_to = conditions[segment_index + 1]

        from_cn = check_condition(cond_from)
        to_cn = check_condition(cond_to)

        if from_cn and to_cn:
            raise ValueError(
                "Interpolating conditions cannot both contain ControlNets"
            )

        try:
            interpolated_condition = [
                (1.0 - local_blend) * c_from + local_blend * c_to
                for c_from, c_to in zip(
                    cond_from[0][0], cond_to[0][0], strict=False
                )
            ]
        except Exception as e:
            print(f"Error during interpolation: {e}")
            raise

        pooled_from = cond_from[0][1].get(
            "pooled_output",
            torch.zeros_like(
                next(iter(cond_from[0][1].values()), torch.tensor([]))
            ),
        )

        pooled_to = cond_to[0][1].get(
            "pooled_output",
            torch.zeros_like(
                next(iter(cond_from[0][1].values()), torch.tensor([]))
            ),
        )

        interpolated_pooled = (
            1.0 - local_blend
        ) * pooled_from + local_blend * pooled_to

        res = {"pooled_output": interpolated_pooled}

        if from_cn:
            res["control"] = cond_from[0][1]["control"]
            res["control_apply_to_uncond"] = cond_from[0][1][
                "control_apply_to_uncond"
            ]
        if to_cn:
            res["control"] = cond_to[0][1]["control"]
            res["control_apply_to_uncond"] = cond_to[0][1][
                "control_apply_to_uncond"
            ]

        return ([(torch.stack(interpolated_condition), res)],)


class MTB_InterpolateClipSequential:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_text": ("STRING", {"multiline": True}),
                "text_to_replace": ("STRING", {"default": ""}),
                "clip": ("CLIP",),
                "interpolation_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "interpolate_encodings_sequential"

    CATEGORY = "mtb/conditioning"

    def interpolate_encodings_sequential(
        self,
        base_text,
        text_to_replace,
        clip,
        interpolation_strength,
        **replacements,
    ):
        log.debug(f"Received interpolation_strength: {interpolation_strength}")

        # - Ensure interpolation strength is within [0, 1]
        interpolation_strength = max(0.0, min(1.0, interpolation_strength))

        # - Check if replacements were provided
        if not replacements:
            raise ValueError("At least one replacement should be provided.")

        num_replacements = len(replacements)
        log.debug(f"Number of replacements: {num_replacements}")

        segment_length = 1.0 / num_replacements
        log.debug(f"Calculated segment_length: {segment_length}")

        # - Find the segment that the interpolation_strength falls into
        segment_index = min(
            int(interpolation_strength // segment_length), num_replacements - 1
        )
        log.debug(f"Segment index: {segment_index}")

        # - Calculate the local strength within the segment
        local_strength = (
            interpolation_strength - (segment_index * segment_length)
        ) / segment_length
        log.debug(f"Local strength: {local_strength}")

        # - If it's the first segment, interpolate between base_text and the first replacement
        if segment_index == 0:
            replacement_text = list(replacements.values())[0]
            log.debug("Using the base text a the base blend")
            # -  Start with the base_text condition
            tokens = clip.tokenize(base_text)
            cond_from, pooled_from = clip.encode_from_tokens(
                tokens, return_pooled=True
            )
        else:
            base_replace = list(replacements.values())[segment_index - 1]
            log.debug(f"Using {base_replace} a the base blend")

            # - Start with the base_text condition replaced by the closest replacement
            tokens = clip.tokenize(
                base_text.replace(text_to_replace, base_replace)
            )
            cond_from, pooled_from = clip.encode_from_tokens(
                tokens, return_pooled=True
            )

            replacement_text = list(replacements.values())[segment_index]

        interpolated_text = base_text.replace(
            text_to_replace, replacement_text
        )
        tokens = clip.tokenize(interpolated_text)
        cond_to, pooled_to = clip.encode_from_tokens(
            tokens, return_pooled=True
        )

        # - Linearly interpolate between the two conditions
        interpolated_condition = (
            1.0 - local_strength
        ) * cond_from + local_strength * cond_to
        interpolated_pooled = (
            1.0 - local_strength
        ) * pooled_from + local_strength * pooled_to

        return (
            [[interpolated_condition, {"pooled_output": interpolated_pooled}]],
        )


class MTB_SmartStep:
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


class MTB_StylesLoader:
    """Load csv files and populate a dropdown from the rows (à la A111)"""

    options = {}

    @classmethod
    def INPUT_TYPES(cls):
        if not cls.options:
            input_dir = Path(folder_paths.base_path) / "styles"
            if not input_dir.exists():
                install_default_styles()

            if not (
                files := [f for f in input_dir.iterdir() if f.suffix == ".csv"]
            ):
                log.warn(
                    "No styles found in the styles folder, place at least one csv file in the styles folder at the root of ComfyUI (for instance ComfyUI/styles/mystyle.csv)"
                )

            for file in files:
                with open(file, encoding="utf8") as f:
                    parsed = csv.reader(f)
                    for i, row in enumerate(parsed):
                        # log.debug(f"Adding style {row[0]}")
                        try:
                            name, positive, negative = (row + [None] * 3)[:3]
                            positive = positive or ""
                            negative = negative or ""
                            if name is not None:
                                cls.options[name] = (positive, negative)
                            else:
                                # Handle the case where 'name' is None
                                log.warning(f"Missing 'name' in row {i}.")

                        except Exception as e:
                            log.warning(
                                f"There was an error while parsing {file}, make sure it respects A1111 format, i.e 3 columns name, positive, negative:\n{e}"
                            )
                            continue

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


__nodes__ = [
    MTB_SmartStep,
    MTB_StylesLoader,
    MTB_InterpolateClipSequential,
    MTB_InterpolateCondition,
]
