from ..utils import tensor2pil
from ..log import log
import io, base64
import torch


class Debug:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"anything_1": ("*")},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "do_debug"
    CATEGORY = "mtb/debug"
    OUTPUT_NODE = True

    def do_debug(self, **kwargs):
        output = {
            "ui": {"b64_images": [], "text": []},
            "result": ("A"),
        }
        for k, v in kwargs.items():
            anything = v
            text = ""
            if isinstance(anything, torch.Tensor):
                log.debug(f"Tensor: {anything.shape}")

                # write the images to temp

                image = tensor2pil(anything)
                b64_imgs = []
                for im in image:
                    buffered = io.BytesIO()
                    im.save(buffered, format="JPEG")
                    b64_imgs.append(
                        "data:image/jpeg;base64,"
                        + base64.b64encode(buffered.getvalue()).decode("utf-8")
                    )

                output["ui"]["b64_images"] += b64_imgs
                log.debug(f"Input {k} contains {len(b64_imgs)} images")
            elif isinstance(anything, bool):
                log.debug(f"Input {k} contains boolean: {anything}")
                output["ui"]["text"] += ["True" if anything else "False"]
            else:
                text = str(anything)
                log.debug(f"Input {k} contains text: {text}")
                output["ui"]["text"] += [text]

        return output


__nodes__ = [Debug]
