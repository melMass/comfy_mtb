import os
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image

from ..log import log


class ImageH264Compression:
    """Encodes the input with h264 compression using a configurable CRF."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The input image tensor to be compressed and decompressed."
                    },
                ),
                "crf": (
                    "INT",
                    {
                        "default": 23,
                        "min": 0,
                        "max": 51,
                        "step": 1,
                        "tooltip": "Constant Rate Factor for h264 encoding (lower values mean higher quality).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compress_and_decompress"

    CATEGORY = "image"
    DESCRIPTION = """
**Encodes the input with h264 compression using a configurable CRF**.

> [!IMPORTANT]
> This node is not really needed with the latest version of LTXVideo.

> [!NOTE]
> This was recommended by the creators of LTX over banodoco's discord.

*Orginal code from [mix](https://github.com/XmYx)*"""

    def _compress_decompress_ffmpeg(self, img_array, crf):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.png")
            output_path = os.path.join(temp_dir, "output.mp4")
            decoded_path = os.path.join(temp_dir, "decoded.png")

            Image.fromarray(img_array).save(input_path)

            encode_command = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-frames:v",
                "1",
                output_path,
            ]
            subprocess.run(encode_command, capture_output=True)

            decode_command = [
                "ffmpeg",
                "-y",
                "-i",
                output_path,
                "-frames:v",
                "1",
                decoded_path,
            ]
            subprocess.run(decode_command, capture_output=True)

            decoded_img = np.array(Image.open(decoded_path))
            return decoded_img

    def compress_and_decompress(self, image, crf):
        import io

        output_images = []

        try:
            import av

            for img_tensor in image:
                img_array = img_tensor.cpu().numpy()
                img_array = (img_array * 255).astype(np.uint8)
                img_array = img_array.copy(
                    order="C"
                )  # Ensure contiguous array

                output = io.BytesIO()

                # Encode the image to h264 with the given CRF
                container = av.open(output, mode="w", format="mp4")
                stream = container.add_stream("h264", rate=1)
                stream.width = img_array.shape[1]
                stream.height = img_array.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": str(crf)}

                frame = av.VideoFrame.from_ndarray(img_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
                container.close()

                # Decode the video back to an image
                output.seek(0)
                container = av.open(output, mode="r", format="mp4")
                decoded_frames = []
                for frame in container.decode(video=0):
                    img_decoded = frame.to_ndarray(format="rgb24")
                    decoded_frames.append(img_decoded)
                container.close()

                if len(decoded_frames) > 0:
                    img_decoded = decoded_frames[0]
                    img_decoded = torch.from_numpy(
                        img_decoded.astype(np.float32) / 255.0
                    )
                    output_images.append(img_decoded)
                else:
                    # If decoding failed, use the original image
                    output_images.append(img_tensor)
        except ImportError:
            log.warning(
                "PyAv is not installed... Falling back to the ffmpeg cli"
            )
            for img_tensor in image:
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                decoded_img = self._compress_decompress_ffmpeg(img_array, crf)
                img_decoded = torch.from_numpy(
                    decoded_img.astype(np.float32) / 255.0
                )
                output_images.append(img_decoded)

        output_images = torch.stack(output_images).to(image.device)
        return (output_images,)


# fmt: off
__nodes__ = [
    ImageH264Compression
]
