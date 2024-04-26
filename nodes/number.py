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


__nodes__ = [
    MTB_IntToBool,
]
