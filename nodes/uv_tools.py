import torch

from ..utils import create_uv_map_tensor, log


class oldDistortImageWithUv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "uv_map": ("UV_MAP",),
                "strength": ("FLOAT", {"default": 1.0, "step": 0.05}),
            },
            "optional": {
                "base_uv_map": ("UV_MAP",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "distort_image_with_uv"
    CATEGORY = "mtb/uv"

    def distort_image_with_uv(
        self, image, uv_map, strength=1.0, base_uv_map=None
    ):
        assert (
            image.shape[1:3] == uv_map.shape[1:3]
        ), "Spatial dimensions of image and uv_map must match!"

        if base_uv_map is None:
            base_uv_map = create_uv_map_tensor(image.shape[2], image.shape[1])

        # Interpolate (or extrapolate) between base UV map and the distorted UV map based on strength
        uv_map = strength * uv_map + (1.0 - strength) * base_uv_map
        # Ensure the image and uv_map have the same spatial dimensions

        # Extract U and V coordinates
        U = uv_map[:, :, :, 0]
        V = uv_map[:, :, :, 1]

        # Convert U and V to pixel coordinates
        b, h, w, _ = image.shape
        U = U * (w - 1)
        V = V * (h - 1)

        # Calculate the four corner indices for each UV coordinate
        U0 = torch.floor(U).long()
        V0 = torch.floor(V).long()
        U1 = U0 + 1
        V1 = V0 + 1

        # Clip the indices to be within the image dimensions
        U0 = torch.clamp(U0, 0, w - 1)
        U1 = torch.clamp(U1, 0, w - 1)
        V0 = torch.clamp(V0, 0, h - 1)
        V1 = torch.clamp(V1, 0, h - 1)

        # Bilinear interpolation weights
        w_U0 = (U1.float() - U).unsqueeze(-1)
        w_U1 = (U - U0.float()).unsqueeze(-1)
        w_V0 = (V1.float() - V).unsqueeze(-1)
        w_V1 = (V - V0.float()).unsqueeze(-1)

        # Sample image using bilinear interpolation
        distorted = (
            (w_U0 * w_V0) * image[:, V0, U0]
            + (w_U0 * w_V1) * image[:, V1, U0]
            + (w_U1 * w_V0) * image[:, V0, U1]
            + (w_U1 * w_V1) * image[:, V1, U1]
        )

        return (distorted.squeeze(0),)


class ImageDistortWithUv:
    """Distorts an image based on a UV map."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "uv_map": ("UV_MAP",),
                "boundary_mode": (
                    ["clamp", "wrap", "reflect", "replicate"],
                    {"default": "wrap"},
                ),
                "strength": ("FLOAT", {"default": 1.0, "step": 0.05}),
            },
            "optional": {
                "base_uv_map": ("UV_MAP",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "distort_image_with_uv"
    CATEGORY = "mtb/uv"

    def distort_image_with_uv(
        self,
        image,
        uv_map,
        boundary_mode="wrap",
        strength=1.0,
        base_uv_map=None,
    ):
        log.debug(f"[UV Distort] Input image shape {image.shape}")
        if image.size(0) == 0:
            log.debug("Input image is empty, returning empty image")
            return (torch.zeros(0),)
        b, h, w, _ = image.shape

        x = w - 1
        y = h - 1

        # If no base UV map provided, create a default one
        if base_uv_map is None:
            base_uv_map = create_uv_map_tensor(w, h).to(image.device)

        # Extract U and V coordinates from the base UV map
        base_U = base_uv_map[..., 0] * x
        base_V = base_uv_map[..., 1] * y

        # Extract U and V coordinates from the distortion UV map and apply strength
        U = strength * uv_map[..., 0] * x + (1 - strength) * base_U
        V = strength * uv_map[..., 1] * y + (1 - strength) * base_V

        # Handle boundary conditions
        if boundary_mode == "wrap":
            U = U % w
            V = V % h
        elif boundary_mode == "reflect":
            U = U % (2 * x)
            V = V % (2 * y)
            U = torch.where(w < U, 2 * x - U, U)
            V = torch.where(h < V, 2 * y - V, V)
        elif boundary_mode == "replicate":
            U = torch.clamp(U, 0, x)
            V = torch.clamp(V, 0, y)
        elif boundary_mode == "clamp":
            U = torch.clamp(U, 0, w)
            V = torch.clamp(V, 0, h)
        else:
            raise ValueError("Invalid boundary_mode")

        # Check if any UV coordinates are out of bounds and log
        if torch.any(w <= U) or torch.any(h <= V):
            log.info("Input UVs out of bounds, clipping")

        # Calculate the four corner indices for each UV coordinate
        U0, V0 = torch.floor(U).long(), torch.floor(V).long()
        # For replicate mode, if U0/V0 is at the last pixel, we replicate that pixel for U1/V1
        if boundary_mode == "replicate":
            U1 = torch.where(x > U0, U0 + 1, U0)
            V1 = torch.where(y > V0, V0 + 1, V0)
        else:
            U1, V1 = U0 + 1, V0 + 1

        # Ensure U1, V1 do not go out of bounds
        U1 = torch.clamp(U1, 0, x)
        V1 = torch.clamp(V1, 0, y)

        # Adjust the bilinear coordinates based on the boundary mode
        if boundary_mode == "wrap":
            U1 = U1 % w
            V1 = V1 % h
        elif boundary_mode == "reflect":
            # This remains unchanged as the coordinates are already reflected above
            pass
        elif boundary_mode == "replicate":
            U1 = torch.clamp(U1, 0, x)
            V1 = torch.clamp(V1, 0, y)

        # Bilinear interpolation weights
        w_U0, w_U1 = (
            (U1.float() - U).unsqueeze(-1),
            (U - U0.float()).unsqueeze(-1),
        )
        w_V0, w_V1 = (
            (V1.float() - V).unsqueeze(-1),
            (V - V0.float()).unsqueeze(-1),
        )

        # Sample image using bilinear interpolation
        distorted = (
            (w_U0 * w_V0) * image[:, V0, U0]
            + (w_U0 * w_V1) * image[:, V1, U0]
            + (w_U1 * w_V0) * image[:, V0, U1]
            + (w_U1 * w_V1) * image[:, V1, U1]
        )

        return (distorted.squeeze(0),)


class UvToImage:
    """Converts the UV map to an image. (Shallow converter)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uv_map": ("UV_MAP",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "uv_to_image"
    CATEGORY = "mtb/uv"

    def uv_to_image(self, uv_map):
        return (uv_map,)


class UvRemoveSeams:
    """Blends values near the UV borders to mitigate visible seams."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uv_map": ("UV_MAP",),
                "radius": ("FLOAT", {"default": 0.01, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("UV_MAP",)
    RETURN_NAMES = ("uv_map",)
    FUNCTION = "remove_uv_seams"
    CATEGORY = "mtb/uv"

    def remove_uv_seams(self, uv_map, radius):
        # Create masks for U and V coordinates close to 0 or 1
        u_border_mask = (uv_map[..., 0] < radius) | (
            uv_map[..., 0] > 1 - radius
        )
        v_border_mask = (uv_map[..., 1] < radius) | (
            uv_map[..., 1] > 1 - radius
        )

        # Soften the UV coordinates near the borders
        uv_map[..., 0] = torch.where(
            u_border_mask, uv_map[..., 0] * 0.5, uv_map[..., 0]
        )
        uv_map[..., 1] = torch.where(
            v_border_mask, uv_map[..., 1] * 0.5, uv_map[..., 1]
        )

        return (uv_map,)


class UvTile:
    """Tiles the UV map based on the specified number of tiles."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uv_map": ("UV_MAP",),
                "tiles_u": ("INT", {"default": 1}),
                "tiles_v": ("INT", {"default": 1}),
                "alt_method": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("UV_MAP",)
    RETURN_NAMES = ("uv_map",)
    FUNCTION = "tile"
    CATEGORY = "mtb/uv"

    def tile(self, uv_map, tiles_u, tiles_v, alt_method=False):
        tiled_uv = uv_map.clone()

        if alt_method:
            tiled_uv[..., 0] = (
                uv_map[..., 0] * tiles_u
            ).floor() / tiles_u + uv_map[..., 0] % (1.0 / tiles_u)
            tiled_uv[..., 1] = (
                uv_map[..., 1] * tiles_v
            ).floor() / tiles_v + uv_map[..., 1] % (1.0 / tiles_v)

        else:
            tiled_uv[..., 0] = (
                uv_map[..., 0] * tiles_u % 1.0
            )  # tile and wrap U coordinates
            tiled_uv[..., 1] = (
                uv_map[..., 1] * tiles_v % 1.0
            )  # tile and wrap V coordinates

        return (tiled_uv,)


class ImageToUv:
    """Turn an image back into a UV map. (Shallow converter)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_uv": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("UV_MAP",)
    RETURN_NAMES = ("uv_map",)
    FUNCTION = "image_to_uv"
    CATEGORY = "mtb/uv"

    def image_to_uv(self, image_uv):
        return (image_uv,)


class UvDistort:
    """Applies a polar coordinates or wave distortion to the UV map"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uv_map": ("UV_MAP",),
                "mode": (["polar", "wave"], {"default": "polar"}),
                "polar_strength": (
                    "FLOAT",
                    {"default": 1.0, "step": 0.05, "min": -1.0, "max": 1.0},
                ),
                "wave_frequency": ("FLOAT", {"default": 10.0}),
                "wave_amplitude": (
                    "FLOAT",
                    {"default": 0.05, "step": 0.05, "min": -1.0, "max": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("UV_MAP",)
    RETURN_NAMES = ("uv_map",)
    FUNCTION = "distort_uvs"
    CATEGORY = "mtb/uv"

    def distort_uvs(
        self,
        uv_map: torch.Tensor,
        mode,
        polar_strength,
        wave_frequency,
        wave_amplitude,
    ):
        if mode == "polar":
            return (self.apply_polar_distortion(uv_map, polar_strength),)
        elif mode == "wave":
            return (
                self.apply_wave_distortion(
                    uv_map, wave_frequency, wave_amplitude
                ),
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

    @classmethod
    def apply_wave_distortion(cls, uv_map, frequency=10.0, amplitude=0.05):
        """
        Applies a wave distortion to the UV map and returns an RGB representation.

        Args:
        - uv_map (torch.Tensor): The UV map tensor.
        - frequency (float): Frequency of the wave.
        - amplitude (float): Amplitude of the wave.

        Returns
        -------
        - torch.Tensor: Distorted UV map in RGB format.
        """
        U = uv_map[:, :, :, 0]
        V = uv_map[:, :, :, 1]

        # Apply wave distortion
        V_distorted = V + amplitude * torch.sin(U * frequency * 2 * 3.14159)

        # Clip V values to [0, 1]
        V_distorted = torch.clamp(V_distorted, 0, 1)

        R = U
        G = V_distorted
        B = torch.zeros_like(R)

        return torch.stack([R, G, B], dim=-1)

    @classmethod
    def apply_polar_distortion(cls, uv_map: torch.Tensor, strength=1.0):
        """
        Applies a polar coordinates distortion to the UV map and returns an RGB representation.

        Args:
        - uv_map (torch.Tensor): The UV map tensor.
        - strength (float): The strength of the distortion.

        Returns
        -------
        - torch.Tensor: Distorted UV map in RGB format.
        """
        U = uv_map[:, :, :, 0]
        V = uv_map[:, :, :, 1]

        # Convert U and V to centered coordinates [-0.5, 0.5]
        U = U * 2 - 1
        V = V * 2 - 1

        # Convert to polar coordinates
        R = torch.sqrt(U * U + V * V)
        Theta = torch.atan2(V, U)

        # Distort the radius
        R_distorted = (
            R + (1.0 - R) * strength
        )  # Changing this line for intuitive strength

        # Convert back to Cartesian
        U_distorted = R_distorted * torch.cos(Theta)
        V_distorted = R_distorted * torch.sin(Theta)

        # Normalize to [0, 1]
        U_distorted = (U_distorted + 1) / 2
        V_distorted = (V_distorted + 1) / 2

        # Clip to ensure values are in [0, 1]
        U_distorted = torch.clamp(U_distorted, 0, 1)
        V_distorted = torch.clamp(V_distorted, 0, 1)

        R = U_distorted
        G = V_distorted
        B = torch.zeros_like(R)

        return torch.stack([R, G, B], dim=-1)


__nodes__ = [
    UvDistort,
    UvToImage,
    ImageToUv,
    ImageDistortWithUv,
    UvTile,
    UvRemoveSeams,
]
