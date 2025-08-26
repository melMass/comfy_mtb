import comfy.sd
import comfy.utils
import folder_paths


class MTB_LoraLoaderModelOnlyByPath:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        all_loras = folder_paths.get_filename_list("loras")

        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (
                    "STRING",
                    {"default": next(iter(all_loras), "no lora found")},
                ),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)

    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    CATEGORY = "mtb/lora"
    FUNCTION = "load_lora_model_only"
    DESCRIPTION = "Exact copy of the native node using string instead of combo, useful to 'build' paths from the graph."

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0)[0],)

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        # all_loras = folder_paths.get_filename_list("loras")
        # if lora_name not in all_loras:
        # raise ValueError(f"{lora_name} not in {folder_paths.get_filename_list('loras')}")

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


__nodes__ = [MTB_LoraLoaderModelOnlyByPath]
