class MTB_PromptPresets:
    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "enprompt"
    CATEGORY = "mtb/prompt"
    EXPERIMENTAL = True

    _all_prompts = {}

    @classmethod
    def INPUT_TYPES(cls):
        from ..utils import comfy_dir
        import json

        cls._all_prompts = {}
        prompt_dir = comfy_dir / "prompts"
        if prompt_dir.exists():
            json_files = prompt_dir.glob("*.json")
            # merge them
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'name' in item and 'prompts' in item:
                                    name = item['name']
                                    prompts = item['prompts']
                                    if isinstance(prompts, list):
                                        cls._all_prompts[name] = prompts
                except Exception as e:
                    print(f"Error loading prompts from {json_file}: {e}")
        return {
            "required": {
                "name": (list(cls._all_prompts.keys()) if cls._all_prompts else ["No prompts found"],),
                "size": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                "random": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    def enprompt(self, name, size, random, seed=None):
        import random as rand

        if name not in self._all_prompts:
            return ([],)

        prompts = self._all_prompts[name].copy()

        if random:
            if seed is not None or seed != -1:
                rand.seed(seed)
            else:
                import time
                rand.seed(int(time.time() * 1000000) % 0xffffffffffffffff)
            rand.shuffle(prompts)

        # Return all prompts
        if size == -1:
            return (prompts,)
        else:
            clamped_size = min(size, len(prompts))
            return (prompts[:clamped_size],)


__nodes__ = [MTB_PromptPresets]
