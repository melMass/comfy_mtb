import comfy.utils
import torch
import torch.nn.functional as F

from ..log import log


class MTB_SceneCutDetector:
    """Detects scene cuts in a video using various methods (content, histogram, hash, or adaptive)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {"tooltip": "The frames used for processing"},
                ),
                "method": (
                    ["content", "histogram", "hash", "adaptive"],
                    {
                        "default": "histogram",
                        "tooltip": "only histogram works properly for now",
                    },
                ),
                "downsample": (
                    ["0.1x", "0.25x", "0.5x", "0.75x", "1.0x"],
                    {
                        "default": "0.1x",
                        "tooltip": "Downsample 'frames' (only for processing)",
                    },
                ),
                "min_scene_length": (
                    "INT",
                    {
                        "default": 15,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "the minimum number of frames a cut can be",
                    },
                ),
                # content
                "content_threshold": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                # histogram
                "histogram_threshold": (
                    "FLOAT",
                    {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "histogram_bins": (
                    "INT",
                    {"default": 128, "min": 2, "max": 256},
                ),
                # hash
                "hash_threshold": (
                    "FLOAT",
                    {"default": 0.395, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "hash_size": ("INT", {"default": 16, "min": 8, "max": 64}),
                # adaptive
                "adaptive_threshold": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.001},
                ),
                "window_width": ("INT", {"default": 2, "min": 1, "max": 10}),
                "min_content_val": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.0, "max": 100.0},
                ),
            },
            "optional": {
                "original_frames": (
                    "IMAGE",
                    {
                        "tooltip": "If provided the returned list will use these frames."
                    },
                ),
            },
        }

    FUNCTION = "detect_cuts"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sequences",)
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "mtb/video"

    def detect_cuts(
        self,
        frames: torch.Tensor,
        method: str,
        min_scene_length: int,
        content_threshold: float = 27.0,
        histogram_threshold: float = 0.05,
        histogram_bins: int = 64,
        hash_threshold: float = 0.395,
        hash_size: int = 16,
        adaptive_threshold: float = 3.0,
        window_width: int = 2,
        min_content_val: float = 15.0,
        downsample: str = "1.0x",
        original_frames: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor]]:
        processing_frames = frames
        frames_to_split = (
            original_frames if original_frames is not None else frames
        )

        if downsample != "1.0x":
            scale = float(downsample.replace("x", ""))
            h, w = frames.shape[1:3]
            new_h, new_w = int(h * scale), int(w * scale)
            processing_frames = F.interpolate(
                frames.permute(0, 3, 1, 2),  # [B,C,H,W] for interpolate
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)  # Back to [B,H,W,C]

        cuts = []
        if method == "content":
            cuts = self.detect_content_cuts(
                processing_frames, content_threshold, min_scene_length
            )
        elif method == "histogram":
            cuts = self.detect_histogram_cuts(
                processing_frames,
                histogram_threshold,
                histogram_bins,
                min_scene_length,
            )
        elif method == "hash":
            cuts = self.detect_hash_cuts(
                processing_frames, hash_threshold, hash_size, min_scene_length
            )
        elif method == "adaptive":
            cuts = self.detect_adaptive_cuts(
                processing_frames,
                adaptive_threshold,
                window_width,
                min_content_val,
                min_scene_length,
            )

        # always include end
        cuts.append(frames.shape[0])

        # split into list
        sequences = [
            frames_to_split[cuts[i] : cuts[i + 1]]
            for i in range(len(cuts) - 1)
        ]
        log.debug(f"Found {len(sequences)} cuts")
        return (sequences,)

    def detect_content_cuts(
        self,
        frames: torch.Tensor,
        threshold: float,
        min_scene_length: int,
    ) -> list[int]:
        """Content-based cut detection using frame differences"""
        num_frames = frames.shape[0]
        device = frames.device
        cuts = [0]
        last_cut = 0

        total = (
            max(0, (num_frames - min_scene_length) - min_scene_length)
            + num_frames
        )

        pbar = comfy.utils.ProgressBar(total)

        differences = torch.zeros(num_frames - 1, device=device)
        for i in range(num_frames - 1):
            differences[i] = self.compute_content_difference(
                frames[i], frames[i + 1]
            )
            pbar.update(1)

        # temporal smoothing
        kernel_size = 3
        differences = F.pad(
            differences.unsqueeze(0).unsqueeze(0),
            ((kernel_size - 1) // 2, (kernel_size - 1) // 2),
            mode="replicate",
        )
        differences = F.avg_pool1d(
            differences, kernel_size, stride=1
        ).squeeze()

        for i in range(min_scene_length, num_frames - min_scene_length):
            pbar.update(1)
            if i - last_cut >= min_scene_length and differences[i] > threshold:
                cuts.append(i)
                last_cut = i

        return cuts

    def detect_histogram_cuts(
        self,
        frames: torch.Tensor,
        threshold: float,
        bins: int,
        min_scene_length: int,
    ) -> list[int]:
        """Histogram-based cut detection"""
        num_frames = frames.shape[0]
        # device = frames.device
        cuts = [0]
        last_cut = 0

        pbar = comfy.utils.ProgressBar(num_frames)

        for i in range(1, num_frames):
            pbar.update(1)
            if i - last_cut < min_scene_length:
                continue

            # Convert to YUV and get Y channel
            yuv1 = (
                0.299 * frames[i - 1, ..., 0]
                + 0.587 * frames[i - 1, ..., 1]
                + 0.114 * frames[i - 1, ..., 2]
            )
            yuv2 = (
                0.299 * frames[i, ..., 0]
                + 0.587 * frames[i, ..., 1]
                + 0.114 * frames[i, ..., 2]
            )

            # Compute histograms
            hist1 = torch.histc(yuv1, bins=bins, min=0, max=1)
            hist2 = torch.histc(yuv2, bins=bins, min=0, max=1)

            # Normalize histograms
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()

            # Compute histogram difference
            diff = torch.sum(torch.abs(hist1 - hist2))

            if diff > threshold:
                cuts.append(i)
                last_cut = i

        return cuts

    def detect_hash_cuts(
        self,
        frames: torch.Tensor,
        threshold: float,
        hash_size: int,
        min_scene_length: int,
    ) -> list[int]:
        """Perceptual hash based cut detection"""
        num_frames = frames.shape[0]
        # device = frames.device
        cuts = [0]
        last_cut = 0

        pbar = comfy.utils.ProgressBar(num_frames)

        def compute_frame_hash(frame):
            # Convert to grayscale
            gray = (
                0.299 * frame[..., 0]
                + 0.587 * frame[..., 1]
                + 0.114 * frame[..., 2]
            )

            gray = F.interpolate(
                gray.unsqueeze(0).unsqueeze(0),
                size=(hash_size, hash_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            dct = torch.fft.rfft2(gray)
            dct = dct[: hash_size // 2, : hash_size // 2]
            return dct > dct.median()

        for i in range(1, num_frames):
            pbar.update(1)
            if i - last_cut < min_scene_length:
                continue

            hash1 = compute_frame_hash(frames[i - 1])
            hash2 = compute_frame_hash(frames[i])

            diff = torch.mean((hash1 != hash2).float())

            if diff > threshold:
                cuts.append(i)
                last_cut = i

        return cuts

    def detect_adaptive_cuts(
        self,
        frames: torch.Tensor,
        adaptive_threshold: float,
        window_width: int,
        min_content_val: float,
        min_scene_length: int,
    ) -> list[int]:
        """Adaptive threshold based cut detection"""
        num_frames = frames.shape[0]
        device = frames.device
        cuts = [0]
        last_cut = 0
        total = num_frames + max(0, (num_frames - window_width) - window_width)

        pbar = comfy.utils.ProgressBar(total)

        content_vals = torch.zeros(num_frames - 1, device=device)
        for i in range(num_frames - 1):
            content_vals[i] = self.compute_content_difference(
                frames[i], frames[i + 1]
            )
            pbar.update(1)

        for i in range(window_width, num_frames - window_width):
            pbar.update(1)
            if i - last_cut < min_scene_length:
                continue

            target_score = content_vals[i]
            window_scores = content_vals[
                i - window_width : i + window_width + 1
            ]
            surrounding_scores = torch.cat(
                [
                    window_scores[:window_width],
                    window_scores[window_width + 1 :],
                ]
            )
            average_score = surrounding_scores.mean()
            if average_score > 1e-5:
                adaptive_ratio = min(target_score / average_score, 255.0)
            elif target_score >= min_content_val:
                adaptive_ratio = 255.0
            else:
                adaptive_ratio = 0.0

            if (
                adaptive_ratio >= adaptive_threshold
                and target_score >= min_content_val
            ):
                cuts.append(i)
                last_cut = i

        return cuts

    def compute_content_difference(
        self, frame1: torch.Tensor, frame2: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes content difference between frames using multiple metrics:
        - Structural similarity
        - Color distribution changes
        - Edge differences
        """
        device = frame1.device

        if frame1.dtype != torch.float32:
            frame1 = frame1.float()
            frame2 = frame2.float()

        def ssim(x, y):
            c1, c2 = 0.01**2, 0.03**2
            mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
            mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)

            mu_x_sq = mu_x.pow(2)
            mu_y_sq = mu_y.pow(2)
            mu_xy = mu_x * mu_y

            sigma_x = (
                F.avg_pool2d(x.pow(2), kernel_size=11, stride=1, padding=5)
                - mu_x_sq
            )
            sigma_y = (
                F.avg_pool2d(y.pow(2), kernel_size=11, stride=1, padding=5)
                - mu_y_sq
            )
            sigma_xy = (
                F.avg_pool2d(x * y, kernel_size=11, stride=1, padding=5)
                - mu_xy
            )

            ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
                (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
            )
            return 1 - ssim_map.mean()

        def color_change(x, y):
            bins = 64
            x_hist = torch.stack(
                [
                    torch.histc(x[..., i], bins=bins, min=0, max=1)
                    for i in range(3)
                ]
            )
            y_hist = torch.stack(
                [
                    torch.histc(y[..., i], bins=bins, min=0, max=1)
                    for i in range(3)
                ]
            )

            x_hist = x_hist / x_hist.sum(dim=1, keepdim=True).clamp(min=1e-6)
            y_hist = y_hist / y_hist.sum(dim=1, keepdim=True).clamp(min=1e-6)

            return torch.mean(torch.abs(x_hist - y_hist))

        def edge_change(x, y):
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device
            ).float()
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device
            ).float()

            def detect_edges(img):
                gray = (
                    0.2989 * img[..., 0]
                    + 0.5870 * img[..., 1]
                    + 0.1140 * img[..., 2]
                )
                gray = gray.unsqueeze(0).unsqueeze(0)

                gx = F.conv2d(gray, sobel_x.view(1, 1, 3, 3), padding=1)
                gy = F.conv2d(gray, sobel_y.view(1, 1, 3, 3), padding=1)

                return torch.sqrt(gx.pow(2) + gy.pow(2)).squeeze()

            edges1 = detect_edges(frame1)
            edges2 = detect_edges(frame2)
            return torch.mean(torch.abs(edges1 - edges2))

        struct_diff = ssim(frame1, frame2)
        color_diff = color_change(frame1, frame2)
        edge_diff = edge_change(frame1, frame2)

        weights = torch.tensor([0.4, 0.3, 0.3], device=device)
        combined_diff = (
            weights[0] * struct_diff
            + weights[1] * color_diff
            + weights[2] * edge_diff
        )

        return combined_diff


__nodes__ = [MTB_SceneCutDetector]
