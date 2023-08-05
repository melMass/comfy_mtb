from ..log import log


class StringReplace:
    """Basic string replacement"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
                "old": ("STRING", {"default": ""}),
                "new": ("STRING", {"default": ""}),
            }
        }

    FUNCTION = "replace_str"
    RETURN_TYPES = ("STRING",)
    CATEGORY = "mtb/string"

    def replace_str(self, string: str, old: str, new: str):
        log.debug(f"Current string: {string}")
        log.debug(f"Find string: {old}")
        log.debug(f"Replace string: {new}")

        string = string.replace(old, new)

        log.debug(f"New string: {string}")

        return (string,)


class FitNumber:
    """Fit the input float using a source and target range"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0, "forceInput": True}),
                "clamp": ("BOOLEAN", {"default": False}),
                "source_min": ("FLOAT", {"default": 0.0}),
                "source_max": ("FLOAT", {"default": 1.0}),
                "target_min": ("FLOAT", {"default": 0.0}),
                "target_max": ("FLOAT", {"default": 1.0}),
            }
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "mtb/math"

    def set_range(
        self,
        value: float,
        clamp: bool,
        source_min: float,
        source_max: float,
        target_min: float,
        target_max: float,
    ):
        res = target_min + (target_max - target_min) * (value - source_min) / (
            source_max - source_min
        )

        if clamp:
            if target_min > target_max:
                res = max(min(res, target_min), target_max)
            else:
                res = max(min(res, target_max), target_min)

        return (res,)


__nodes__ = [StringReplace, FitNumber]
