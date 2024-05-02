class MTB_IntToBool:
    """Basic int to bool conversion"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": (
                    "INT",
                    {
                        "default": 0,
                    },
                ),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "int_to_bool"
    CATEGORY = "mtb/number"

    def int_to_bool(self, int):
        return (bool(int),)


class MTB_IntToNumber:
    """Node addon for the WAS Suite. Converts a "comfy" INT to a NUMBER."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1e9,
                        "max": 1e9,
                        "step": 1,
                        "forceInput": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("NUMBER",)
    FUNCTION = "int_to_number"
    CATEGORY = "mtb/number"

    def int_to_number(self, int):
        return (int,)


class MTB_FloatToNumber:
    """Node addon for the WAS Suite. Converts a "comfy" FLOAT to a NUMBER."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": -1e9,
                        "max": 1e9,
                        "step": 1,
                        "forceInput": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("NUMBER",)
    FUNCTION = "float_to_number"
    CATEGORY = "mtb/number"

    def float_to_number(self, float):
        return (float,)


__nodes__ = [
    MTB_FloatToNumber,
    MTB_IntToBool,
    MTB_IntToNumber,
]
