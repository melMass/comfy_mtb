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
    ):
        if image.size(0) == 0:
            return (torch.zeros(0),)
        transformed_images = []
        for img in image:
            img = img.transpose(0, 2)

            transformed_image = F.affine(
                img, angle=angle, scale=zoom, translate=[int(y), int(x)], shear=shear
            )

            transformed_image = transformed_image.transpose(2, 0)
            transformed_images.append(transformed_image.unsqueeze(0))

        return (torch.cat(transformed_images, dim=0),)


__nodes__ = [TransformImage]
