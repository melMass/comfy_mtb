from ..log import log


class AnimationBuilder:
    """Convenient way to manage basic animation maths at the core of many of my workflows"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {"default": 100, "min": 0}),
                # "fps": ("INT", {"default": 12, "min": 0}),
                "scale_float": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "loop_count": ("INT", {"default": 1, "min": 0}),
                "raw_iteration": ("INT", {"default": 0, "min": 0}),
                "raw_loop": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("frame", "0-1 (scaled)", "count", "loop_ended")
    CATEGORY = "mtb/animation"
    FUNCTION = "build_animation"

    def build_animation(
        self,
        total_frames=100,
        # fps=12,
        scale_float=1.0,
        loop_count=1,  # set in js
        raw_iteration=0,  # set in js
        raw_loop=0,  # set in js
    ):
        frame = raw_iteration % (total_frames)
        scaled = (frame / (total_frames - 1)) * scale_float
        # if frame == 0:
        #     log.debug("Reseting history")
        #     PromptServer.instance.prompt_queue.wipe_history()
        log.debug(f"frame: {frame}/{total_frames}  scaled: {scaled}")

        return (frame, scaled, raw_loop, (frame == (total_frames - 1)))


__nodes__ = [AnimationBuilder]
