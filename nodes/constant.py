import json

from ..log import log


class MTB_Constant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"Value": ("*",)},
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    CATEGORY = "mtb/utils"
    FUNCTION = "execute"

    def execute(
        self,
        **kwargs,
    ):
        log.debug("Received kwargs")
        log.debug(json.dumps(kwargs, check_circular=True))
        return (kwargs.get("Value"),)


# __nodes__ = [MTB_Constant]
