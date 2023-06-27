class IntToNumber:
    """Node addon for the WAS Suite. Converts a "comfy" INT to a NUMBER."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": ("INT", {"default": 0, "min": 0, "max": 1e9, "step": 1}),
            }
        }

    RETURN_TYPES = ("NUMBER",)
    FUNCTION = "int_to_number"
    CATEGORY = "number"

    def int_to_number(self, int):

        return (int,)

__nodes__ = [
    IntToNumber,

]