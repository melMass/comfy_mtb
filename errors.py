class ModelNotFound(Exception):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(
            f"The model {model_name} could not be found, make sure to download it using ComfyManager first.\nrepository: https://github.com/ltdrdata/ComfyUI-Manager",
            *args,
            **kwargs,
        )
