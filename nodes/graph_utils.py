class Modulo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": ("INT", {"default": 0, "min": 0, "max": 1e9, "step": 1}),
                "mod": ("INT", {"default": 0, "min": 0, "max": 1e9, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "modulo"
    CATEGORY = "number"

    def modulo(self, int, mod):

        return ((int + 1) % (mod + 1),)


class IntToNumber:
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
