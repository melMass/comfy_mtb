from ..log import log


class MTB_AnimationBuilder:
    """Simple maths for animation."""

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
    DESCRIPTION = """
# Animation Builder

Check the
[wiki page](https://github.com/melMass/comfy_mtb/wiki/nodes-animation-builder)
for more info.


- This basic example should help to understand the meaning of
its inputs and outputs thanks to the [debug](nodes-debug) node.

![](https://github.com/melMass/comfy_mtb/assets/7041726/2b5c7e4f-372d-4494-9e73-abb2daa7cb36)

- In this other example Animation Builder is used in combination with
[Batch From History](https://github.com/melMass/comfy_mtb/wiki/nodes-batch-from-history)
to create a zoom-in animation on a static image

![](https://github.com/melMass/comfy_mtb/assets/7041726/77d37da1-0a8e-4519-a493-dfdef7f755ea)

## Inputs

| name | description |
| ---- | :----------:|
| total_frames | The number of frame to queue (this is multiplied by the `loop_count`)|
| scale_float | Convenience input to scale the normalized `current value` (a float between 0 and 1 lerp over the current queue length) |
| loop_count | The number of loops to queue |
| **Reset Button** | resets the internal counters, although the node is though around using its queue button it should still work fine when using the regular queue button of comfy |
| **Queue Button** | Convenience button to run the queues (`total_frames` * `loop_count`) |

"""

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


__nodes__ = [MTB_AnimationBuilder]
