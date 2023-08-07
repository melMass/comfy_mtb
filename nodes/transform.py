import torch
import torchvision.transforms.functional as F


class TransformImage:
    """Save torch tensors (image, mask or latent) to disk, useful to debug things outside comfy


    it return a tensor representing the transformed images with the same shape as the input tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("FLOAT", {"default": 0}),
                "y": ("FLOAT", {"default": 0}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.001}),
                "angle": ("FLOAT", {"default": 0}),
                "shear": ("FLOAT", {"default": 0}),
                "border_handling": (
                    ["edge", "constant", "reflect", "symmetric"],
                    {"default": "edge"},
                ),
            },
        }

    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "mtb/transform"

    def transform(
        self,
        image: torch.Tensor,
        x: float,
        y: float,
        zoom: float,
        angle: int,
        shear,
        border_handling="edge",
    ):
        if image.size(0) == 0:
            return (torch.zeros(0),)
        transformed_images = []
        frames_count, frame_height, frame_width, frame_channel_count = image.size()

        new_height, new_width = int(frame_height * zoom), int(frame_width * zoom)

        pw = int(frame_width - new_width)
        ph = int(frame_height - new_height)
        padding = [max(0, pw + x), max(0, ph + y), max(0, pw - x), max(0, ph - y)]

        for img in image:
            img = img.permute(2, 0, 1)
            new_height, new_width = int(frame_height * zoom), int(frame_width * zoom)
            pw = int(frame_width - new_width)
            ph = int(frame_height - new_height)

            padding = [int(i) for i in padding]

            img = F.pad(
                img,  # transformed_frame,
                padding=padding,
                padding_mode=border_handling,
            )

            img = F.affine(img, angle=angle, scale=zoom, translate=[x, y], shear=shear)

            crop = [ph + y, -(ph - y), x + pw, -(pw - x)]

            img = img[:, crop[0] : crop[1], crop[2] : crop[3]]

            img = img.permute(1, 2, 0)
            transformed_images.append(img.unsqueeze(0))

        return (torch.cat(transformed_images, dim=0),)


__nodes__ = [TransformImage]
