import math
import os
from pathlib import Path
from typing import List

import cv2
import folder_paths
import numpy as np
import torch

from ..log import log
from ..utils import apply_easing, pil2tensor
from .transform import TransformImage


def hex_to_rgb(hex_color, bgr=False):
    hex_color = hex_color.lstrip("#")
    if bgr:
        return tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))

    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


class BatchMake:
    """Simply duplicates the input frame as a batch"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "count": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_batch"
    CATEGORY = "mtb/batch"

    def generate_batch(self, image: torch.Tensor, count):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        return (image.repeat(count, 1, 1, 1),)


class BatchShape:
    """Generates a batch of 2D shapes with optional shading (experimental)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 1}),
                "shape": (
                    ["Box", "Circle", "Diamond"],
                    {"default": "Box"},
                ),
                "image_width": ("INT", {"default": 512}),
                "image_height": ("INT", {"default": 512}),
                "shape_size": ("INT", {"default": 100}),
                "color": ("COLOR", {"default": "#ffffff"}),
                "bg_color": ("COLOR", {"default": "#000000"}),
                "shade_color": ("COLOR", {"default": "#000000"}),
                "shadex": ("FLOAT", {"default": 0.0}),
                "shadey": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_shapes"
    CATEGORY = "mtb/batch"

    def generate_shapes(
        self,
        count,
        shape,
        image_width,
        image_height,
        shape_size,
        color,
        bg_color,
        shade_color,
        shadex,
        shadey,
    ):
        print(f"COLOR: {color}")
        print(f"BG_COLOR: {bg_color}")
        print(f"SHADE_COLOR: {shade_color}")

        # Parse color input to BGR tuple for OpenCV
        color = hex_to_rgb(color)
        bg_color = hex_to_rgb(bg_color)
        shade_color = hex_to_rgb(shade_color)
        res = []
        for x in range(count):
            # Initialize an image canvas
            canvas = np.full((image_height, image_width, 3), bg_color, dtype=np.uint8)
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

            # Compute the center point of the shape
            center = (image_width // 2, image_height // 2)

            if shape == "Box":
                half_size = shape_size // 2
                top_left = (center[0] - half_size, center[1] - half_size)
                bottom_right = (center[0] + half_size, center[1] + half_size)
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)
            elif shape == "Circle":
                cv2.circle(mask, center, shape_size // 2, 255, -1)
            elif shape == "Diamond":
                pts = np.array(
                    [
                        [center[0], center[1] - shape_size // 2],
                        [center[0] + shape_size // 2, center[1]],
                        [center[0], center[1] + shape_size // 2],
                        [center[0] - shape_size // 2, center[1]],
                    ]
                )
                cv2.fillPoly(mask, [pts], 255)

            # Color the shape
            canvas[mask == 255] = color

            # Apply shading effects to a separate shading canvas
            shading = np.zeros_like(canvas, dtype=np.float32)
            shading[:, :, 0] = shadex * np.linspace(0, 1, image_width)
            shading[:, :, 1] = shadey * np.linspace(0, 1, image_height).reshape(-1, 1)
            shading_canvas = cv2.addWeighted(
                canvas.astype(np.float32), 1, shading, 1, 0
            ).astype(np.uint8)

            # Apply shading only to the shape area using the mask
            canvas[mask == 255] = shading_canvas[mask == 255]
            res.append(canvas)

        return (pil2tensor(res),)


class BatchFloatFill:
    """Fills a batch float with a single value until it reaches the target length"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS",),
                "direction": (["head", "tail"], {"default": "tail"}),
                "value": ("FLOAT", {"default": 0.0}),
                "count": ("INT", {"default": 1}),
            }
        }

    FUNCTION = "fill_floats"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"

    def fill_floats(self, floats, direction, value, count):
        size = len(floats)
        if size > count:
            raise ValueError(f"Size ({size}) is less then target count ({count})")

        rem = count - size
        if direction == "tail":
            floats = floats + [value] * rem
        else:
            floats = [value] * rem + floats
        return (floats,)


class BatchFloatAssemble:
    """Assembles mutiple batches of floats into a single stream (batch)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reverse": ("BOOLEAN", {"default": False})}}

    FUNCTION = "assemble_floats"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"

    def assemble_floats(self, reverse, **kwargs):
        res = []
        if reverse:
            for x in reversed(kwargs.values()):
                res += x
        else:
            for x in kwargs.values():
                res += x

        return (res,)


class BatchFloat:
    """Generates a batch of float values with interpolation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["Single", "Steps"],
                    {"default": "Steps"},
                ),
                "count": ("INT", {"default": 1}),
                "min": ("FLOAT", {"default": 0.0}),
                "max": ("FLOAT", {"default": 1.0}),
                "easing": (
                    [
                        "Linear",
                        "Sine In",
                        "Sine Out",
                        "Sine In/Out",
                        "Quart In",
                        "Quart Out",
                        "Quart In/Out",
                        "Cubic In",
                        "Cubic Out",
                        "Cubic In/Out",
                        "Circ In",
                        "Circ Out",
                        "Circ In/Out",
                        "Back In",
                        "Back Out",
                        "Back In/Out",
                        "Elastic In",
                        "Elastic Out",
                        "Elastic In/Out",
                        "Bounce In",
                        "Bounce Out",
                        "Bounce In/Out",
                    ],
                    {"default": "Linear"},
                ),
            }
        }

    FUNCTION = "set_floats"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"

    def set_floats(self, mode, count, min, max, easing):
        keyframes = []
        if mode == "Single":
            keyframes = [min] * count
            return (keyframes,)

        for i in range(count):
            normalized_step = i / (count - 1)
            eased_step = apply_easing(normalized_step, easing)
            eased_value = min + (max - min) * eased_step
            keyframes.append(eased_value)

        return (keyframes,)


class BatchMerge:
    """Merges multiple image batches with different frame counts"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fusion_mode": (["add", "multiply", "average"], {"default": "average"}),
                "fill": (["head", "tail"], {"default": "tail"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge_batches"
    CATEGORY = "mtb/batch"

    def merge_batches(self, fusion_mode, fill, **kwargs):
        images = kwargs.values()
        max_frames = max(img.shape[0] for img in images)

        adjusted_images = []
        for img in images:
            frame_count = img.shape[0]
            if frame_count < max_frames:
                fill_frame = img[0] if fill == "head" else img[-1]
                fill_frames = fill_frame.repeat(max_frames - frame_count, 1, 1, 1)
                adjusted_batch = (
                    torch.cat((fill_frames, img), dim=0)
                    if fill == "head"
                    else torch.cat((img, fill_frames), dim=0)
                )
            else:
                adjusted_batch = img
            adjusted_images.append(adjusted_batch)

        # Merge the adjusted batches
        merged_image = None
        for img in adjusted_images:
            if merged_image is None:
                merged_image = img
            else:
                if fusion_mode == "add":
                    merged_image += img
                elif fusion_mode == "multiply":
                    merged_image *= img
                elif fusion_mode == "average":
                    merged_image = (merged_image + img) / 2

        return (merged_image,)


class Batch2dTransform:
    """Transform a batch of images using a batch of keyframes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_handling": (
                    ["edge", "constant", "reflect", "symmetric"],
                    {"default": "edge"},
                ),
                "constant_color": ("COLOR", {"default": "#000000"}),
            },
            "optional": {
                "x": ("FLOATS",),
                "y": ("FLOATS",),
                "zoom": ("FLOATS",),
                "angle": ("FLOATS",),
                "shear": ("FLOATS",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_batch"
    CATEGORY = "mtb/batch"

    def get_num_elements(self, param) -> int:
        if isinstance(param, torch.Tensor):
            return torch.numel(param)
        elif isinstance(param, list):
            return len(param)

        return 0

    def transform_batch(
        self,
        image: torch.Tensor,
        border_handling,
        constant_color,
        x=None,
        y=None,
        zoom=None,
        angle=None,
        shear=None,
    ):
        if all(
            self.get_num_elements(param) <= 0 for param in [x, y, zoom, angle, shear]
        ):
            raise ValueError("At least one transform parameter must be provided")

        keyframes = {"x": [], "y": [], "zoom": [], "angle": [], "shear": []}

        default_vals = {"x": 0, "y": 0, "zoom": 1.0, "angle": 0, "shear": 0}

        if self.get_num_elements(x) > 0:
            keyframes["x"] = x
        if self.get_num_elements(y) > 0:
            keyframes["y"] = y
        if self.get_num_elements(zoom) > 0:
            keyframes["zoom"] = zoom
        if self.get_num_elements(angle) > 0:
            keyframes["angle"] = angle
        if self.get_num_elements(shear) > 0:
            keyframes["shear"] = shear

        for name, values in keyframes.items():
            count = len(values)
            if count > 0 and count != image.shape[0]:
                raise ValueError(
                    f"Length of {name} values ({count}) must match number of images ({image.shape[0]})"
                )
            if count == 0:
                keyframes[name] = [default_vals[name]] * image.shape[0]

        transformer = TransformImage()
        res = [
            transformer.transform(
                image[i].unsqueeze(0),
                keyframes["x"][i],
                keyframes["y"][i],
                keyframes["zoom"][i],
                keyframes["angle"][i],
                keyframes["shear"][i],
                border_handling,
                constant_color,
            )[0]
            for i in range(image.shape[0])
        ]
        return (torch.cat(res, dim=0),)


DEFAULT_INTERPOLANT = lambda t: t * t * t * (t * (t * 6 - 15) + 10)


class BatchShake:
    """Applies a shaking effect to batches of images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "position_amount_x": ("FLOAT", {"default": 1.0}),
                "position_amount_y": ("FLOAT", {"default": 1.0}),
                "rotation_amount": ("FLOAT", {"default": 10.0}),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.005}),
                "frequency_divider": ("FLOAT", {"default": 1.0, "min": 0.005}),
                "octaves": ("INT", {"default": 1, "min": 1}),
                "seed": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOATS", "FLOATS", "FLOATS")
    RETURN_NAMES = ("image", "pos_x", "pos_y", "rot")
    FUNCTION = "apply_shake"
    CATEGORY = "mtb/batch"

    # def interpolant(self, t):
    # return t * t * t * (t * (t * 6 - 15) + 10)

    def generate_perlin_noise_2d(
        self, shape, res, tileable=(False, False), interpolant=None
    ):
        """Generate a 2D numpy array of perlin noise.

        Args:
            shape: The shape of the generated array (tuple of two ints).
                This must be a multple of res.
            res: The number of periods of noise to generate along each
                axis (tuple of two ints). Note shape must be a multiple of
                res.
            tileable: If the noise should be tileable along each axis
                (tuple of two bools). Defaults to (False, False).
            interpolant: The interpolation function, defaults to
                t*t*t*(t*(t*6 - 15) + 10).

        Returns:
            A numpy array of shape shape with the generated noise.

        Raises:
            ValueError: If shape is not a multiple of res.
        """
        interpolant = interpolant or DEFAULT_INTERPOLANT
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = (
            np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0)
            % 1
        )
        # Gradients
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        if tileable[0]:
            gradients[-1, :] = gradients[0, :]
        if tileable[1]:
            gradients[:, -1] = gradients[:, 0]
        gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
        g00 = gradients[: -d[0], : -d[1]]
        g10 = gradients[d[0] :, : -d[1]]
        g01 = gradients[: -d[0], d[1] :]
        g11 = gradients[d[0] :, d[1] :]
        # Ramps
        n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        # Interpolation
        t = interpolant(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    def generate_fractal_noise_2d(
        self,
        shape,
        res,
        octaves=1,
        persistence=0.5,
        lacunarity=2,
        tileable=(True, True),
        interpolant=None,
    ):
        """Generate a 2D numpy array of fractal noise.

        Args:
            shape: The shape of the generated array (tuple of two ints).
                This must be a multiple of lacunarity**(octaves-1)*res.
            res: The number of periods of noise to generate along each
                axis (tuple of two ints). Note shape must be a multiple of
                (lacunarity**(octaves-1)*res).
            octaves: The number of octaves in the noise. Defaults to 1.
            persistence: The scaling factor between two octaves.
            lacunarity: The frequency factor between two octaves.
            tileable: If the noise should be tileable along each axis
                (tuple of two bools). Defaults to (True,True).
            interpolant: The, interpolation function, defaults to
                t*t*t*(t*(t*6 - 15) + 10).

        Returns:
            A numpy array of fractal noise and of shape shape generated by
            combining several octaves of perlin noise.

        Raises:
            ValueError: If shape is not a multiple of
                (lacunarity**(octaves-1)*res).
        """
        interpolant = interpolant or DEFAULT_INTERPOLANT

        noise = np.zeros(shape)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.generate_perlin_noise_2d(
                shape, (frequency * res[0], frequency * res[1]), tileable, interpolant
            )
            frequency *= lacunarity
            amplitude *= persistence
        return noise

    def fbm(self, x, y, octaves):
        # noise_2d = self.generate_fractal_noise_2d((256, 256), (8, 8), octaves)
        # Now, extract a single noise value based on x and y, wrapping indices if necessary
        x_idx = int(x) % 256
        y_idx = int(y) % 256
        return self.noise_pattern[x_idx, y_idx]

    def apply_shake(
        self,
        images,
        position_amount_x,
        position_amount_y,
        rotation_amount,
        frequency,
        frequency_divider,
        octaves,
        seed,
    ):
        # Rehash
        np.random.seed(seed)
        self.position_offset = np.random.uniform(-1e3, 1e3, 3)
        self.rotation_offset = np.random.uniform(-1e3, 1e3, 3)
        self.noise_pattern = self.generate_perlin_noise_2d(
            (512, 512), (32, 32), (True, True)
        )

        # Assuming frame count is derived from the first dimension of images tensor
        frame_count = images.shape[0]

        frequency = frequency / frequency_divider

        # Generate shaking parameters for each frame
        x_translations = []
        y_translations = []
        rotations = []

        for frame_num in range(frame_count):
            time = frame_num * frequency
            x_idx = (self.position_offset[0] + frame_num) % 256
            y_idx = (self.position_offset[1] + frame_num) % 256

            np_position = np.array(
                [
                    self.fbm(x_idx, time, octaves),
                    self.fbm(y_idx, time, octaves),
                ]
            )

            # np_position = np.array(
            #     [
            #         self.fbm(self.position_offset[0] + frame_num, time, octaves),
            #         self.fbm(self.position_offset[1] + frame_num, time, octaves),
            #     ]
            # )
            # np_rotation = self.fbm(self.rotation_offset[2] + frame_num, time, octaves)

            rot_idx = (self.rotation_offset[2] + frame_num) % 256
            np_rotation = self.fbm(rot_idx, time, octaves)

            x_translations.append(np_position[0] * position_amount_x)
            y_translations.append(np_position[1] * position_amount_y)
            rotations.append(np_rotation * rotation_amount)

        # Convert lists to tensors
        # x_translations = torch.tensor(x_translations, dtype=torch.float32)
        # y_translations = torch.tensor(y_translations, dtype=torch.float32)
        # rotations = torch.tensor(rotations, dtype=torch.float32)

        # Create an instance of Batch2dTransform
        transform = Batch2dTransform()

        log.debug(
            f"Applying shaking with parameters: \nposition {position_amount_x}, {position_amount_y}\nrotation {rotation_amount}\nfrequency {frequency}\noctaves {octaves}"
        )

        # Apply shaking transformations to images
        shaken_images = transform.transform_batch(
            images,
            border_handling="edge",  # Assuming edge handling as default
            constant_color="#000000",  # Assuming black as default constant color
            x=x_translations,
            y=y_translations,
            angle=rotations,
        )[0]

        return (shaken_images, x_translations, y_translations, rotations)


__nodes__ = [
    BatchFloat,
    Batch2dTransform,
    BatchShape,
    BatchMake,
    BatchFloatAssemble,
    BatchFloatFill,
    BatchMerge,
    BatchShake,
]
