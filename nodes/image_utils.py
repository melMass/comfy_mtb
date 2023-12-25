import torch

from ..log import log


class StackImages:
    """Stack the input images horizontally or vertically"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vertical": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stack"
    CATEGORY = "mtb/image utils"

    def stack(self, vertical, **kwargs):
        if not kwargs:
            raise ValueError("At least one tensor must be provided.")

        tensors = list(kwargs.values())
        log.debug(
            f"Stacking {len(tensors)} tensors {'vertically' if vertical else 'horizontally'}"
        )
        log.debug(list(kwargs.keys()))

        ref_shape = tensors[0].shape
        for tensor in tensors[1:]:
            if tensor.shape[1:] != ref_shape[1:]:
                raise ValueError(
                    "All tensors must have the same dimensions except for the stacking dimension."
                )

        dim = 1 if vertical else 2

        stacked_tensor = torch.cat(tensors, dim=dim)

        return (stacked_tensor,)


class PickFromBatch:
    """Pick a specific number of images from a batch, either from the start or end."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "from_direction": (["end", "start"], {"default": "start"}),
                "count": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pick_from_batch"
    CATEGORY = "mtb/image utils"

    def pick_from_batch(self, image, from_direction, count):
        batch_size = image.size(0)

        # Limit count to the available number of images in the batch
        count = min(count, batch_size)
        if count < batch_size:
            log.warning(
                f"Requested {count} images, but only {batch_size} are available."
            )

        if from_direction == "end":
            selected_tensors = image[-count:]
        else:
            selected_tensors = image[:count]

        return (selected_tensors,)


__nodes__ = [StackImages, PickFromBatch]
