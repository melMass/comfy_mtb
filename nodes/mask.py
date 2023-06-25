from rembg import remove
from ..utils import pil2tensor, tensor2pil
from PIL import Image

class ImageRemoveBackgroundRembg:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_matting": (["True","False"], {"default":"False"},),
                "alpha_matting_foreground_threshold": ("INT", {"default":240, "min": 0, "max": 255},),
                "alpha_matting_background_threshold": ("INT", {"default":10, "min": 0, "max": 255},),
                "alpha_matting_erode_size": ("INT", {"default":10, "min": 0, "max": 255},),
                "post_process_mask": (["True","False"], {"default":"False"},),
                "bgcolor": ("COLOR", {"default":"black"},),
                
            },
        }

    RETURN_TYPES = ("IMAGE","MASK","IMAGE",)
    RETURN_NAMES = ("Image (rgba)","Mask","Image",)
    FUNCTION = "remove_background"
    CATEGORY = "image"
    # bgcolor: Optional[Tuple[int, int, int, int]]
    def remove_background(self, image, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size, post_process_mask, bgcolor):
        print(f"Background Color: {bgcolor}")
        image = remove(
                data=tensor2pil(image),
                alpha_matting=alpha_matting == "True",
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                session=None,
                only_mask=False,
                post_process_mask=post_process_mask == "True",
                bgcolor=None
            )
        
        
        # extract the alpha to a new image
        mask = image.getchannel(3)
        
        # add our bgcolor behind the image
        image_on_bg = Image.new("RGBA", image.size, bgcolor)
        
        image_on_bg.paste(image, mask=mask)
        
        
        return (pil2tensor(image), pil2tensor(mask), pil2tensor(image_on_bg))
