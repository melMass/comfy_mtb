import torch
from ..log import log


class StackImages:
    """Stack the input images horizontally or vertically."""

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
            f"Stacking {len(tensors)} tensors "
            f"{'vertically' if vertical else 'horizontally'}"
        )

        normalized_tensors = [
            self.normalize_to_rgba(tensor) for tensor in tensors
        ]

        if vertical:
            width = normalized_tensors[0].shape[2]
            if any(tensor.shape[2] != width for tensor in normalized_tensors):
                raise ValueError(
                    "All tensors must have the same width "
                    "for vertical stacking."
                )
            dim = 1
        else:
            height = normalized_tensors[0].shape[1]
            if any(tensor.shape[1] != height for tensor in normalized_tensors):
                raise ValueError(
                    "All tensors must have the same height "
                    "for horizontal stacking."
                )
            dim = 2

        stacked_tensor = torch.cat(normalized_tensors, dim=dim)

        return (stacked_tensor,)

    def normalize_to_rgba(self, tensor):
        """Normalize tensor to have 4 channels (RGBA)."""
        _, _, _, channels = tensor.shape
        # already RGBA
        if channels == 4:
            return tensor
        # RGB to RGBA
        elif channels == 3:
            alpha_channel = torch.ones(
                tensor.shape[:-1] + (1,), device=tensor.device
            )  # Add an alpha channel
            return torch.cat((tensor, alpha_channel), dim=-1)
        else:
            raise ValueError(
                "Tensor has an unsupported number of channels: "
                "expected 3 (RGB) or 4 (RGBA)."
            )


class PickFromBatch:
    """Pick a specific number of images from a batch.

    either from the start or end.
    """

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
                f"Requested {count} images, "
                f"but only {batch_size} are available."
            )

        if from_direction == "end":
            selected_tensors = image[-count:]
        else:
            selected_tensors = image[:count]

        return (selected_tensors,)


__nodes__ = [StackImages, PickFromBatch]
