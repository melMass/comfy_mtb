class SmartStep:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "step": ("INT",{"default": 20, "min": 1, "max": 10000, "step": 1},),
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

    RETURN_TYPES = ("INT","INT","INT")
    RETURN_NAMES = ("step","start","end")
    FUNCTION = "do_step"
    CATEGORY = "conditioning"

    def do_step(self, step,start_percent,end_percent):
        
        start = int(step * start_percent / 100)
        end = int(step * end_percent / 100)
        
        return (step,start, end)