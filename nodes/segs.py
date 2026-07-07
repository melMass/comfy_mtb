import logging

import numpy as np
import torch


def lerp(v0, v1, t):
    return v0 * (1.0 - t) + v1 * t


def interpolate_box(box1, box2, t):
    """Linearly interpolate between two boxes (tuples of 4 ints)."""
    if box1 is None or box2 is None:
        return None
    x1 = lerp(box1[0], box2[0], t)
    y1 = lerp(box1[1], box2[1], t)
    x2 = lerp(box1[2], box2[2], t)
    y2 = lerp(box1[3], box2[3], t)
    return (int(x1), int(y1), int(x2), int(y2))


class MTB_FaceMeshBatchToSEGSAndFill:
    @classmethod
    def INPUT_TYPES(s):
        bool_true_widget = (
            "BOOLEAN",
            {"default": True, "label_on": "Enabled", "label_off": "Disabled"},
        )
        bool_false_widget = (
            "BOOLEAN",
            {"default": False, "label_on": "Enabled", "label_off": "Disabled"},
        )

        return {
            "required": {
                "image": ("IMAGE",),
                "interpolation": (
                    ["linear", "hold_last"],
                    {"default": "linear"},
                ),
                "fill_shape": (
                    ["interpolate_contour", "bbox"],
                    {"default": "interpolate_contour"},
                ),
                "crop_factor": (
                    "FLOAT",
                    {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1},
                ),
                "bbox_fill": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "enabled",
                        "label_off": "disabled",
                    },
                ),
                "drop_size": (
                    "INT",
                    {"min": 1, "max": 8192, "step": 1, "default": 1},
                ),
                "dilation": (
                    "INT",
                    {"default": 0, "min": -512, "max": 512, "step": 1},
                ),
                "face": bool_true_widget,
                "mouth": bool_false_widget,
                "left_eyebrow": bool_false_widget,
                "left_eye": bool_false_widget,
                "left_pupil": bool_false_widget,
                "right_eyebrow": bool_false_widget,
                "right_eye": bool_false_widget,
                "right_pupil": bool_false_widget,
            }
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "generate_and_fill"
    CATEGORY = "ImpactPack/Operation"

    def generate_and_fill(
        self,
        image,
        interpolation,
        fill_shape,
        crop_factor,
        bbox_fill,
        drop_size,
        dilation,
        face,
        mouth,
        left_eyebrow,
        left_eye,
        left_pupil,
        right_eyebrow,
        right_eye,
        right_pupil,
    ):
        import cv2
        import impact.core as core

        total_frames = image.shape[0]
        logging.info(
            f"[FaceMesh Batch & Fill] Starting. Processing {total_frames} frames. Mode: {interpolation}/{fill_shape}."
        )

        all_labels_found = set()
        structured_segs = {}
        for i in range(total_frames):
            single_frame_batch = image[i : i + 1]
            _, segs_for_this_frame = core.mediapipe_facemesh_to_segs(
                single_frame_batch,
                crop_factor,
                bbox_fill,
                50,
                drop_size,
                dilation,
                face,
                mouth,
                left_eyebrow,
                left_eye,
                left_pupil,
                right_eyebrow,
                right_eye,
                right_pupil,
            )
            for seg in segs_for_this_frame:
                if seg.label not in all_labels_found:
                    all_labels_found.add(seg.label)
                    structured_segs[seg.label] = [None] * total_frames
                structured_segs[seg.label][i] = seg

        for label, timeline in structured_segs.items():
            if interpolation == "linear":
                last_valid_idx = -1
                for i in range(total_frames):
                    if timeline[i] is not None:
                        if i > last_valid_idx + 1 and last_valid_idx != -1:
                            start_seg, end_seg = (
                                timeline[last_valid_idx],
                                timeline[i],
                            )
                            gap_size = i - last_valid_idx
                            for j in range(1, gap_size):
                                t = j / float(gap_size)
                                new_bbox = interpolate_box(
                                    start_seg.bbox, end_seg.bbox, t
                                )
                                new_crop = interpolate_box(
                                    start_seg.crop_region,
                                    end_seg.crop_region,
                                    t,
                                )

                                if new_crop is None:
                                    continue
                                crop_h, crop_w = (
                                    new_crop[3] - new_crop[1],
                                    new_crop[2] - new_crop[0],
                                )
                                if crop_h <= 0 or crop_w <= 0:
                                    continue

                                new_mask = None
                                if fill_shape == "interpolate_contour":
                                    start_mask = start_seg.cropped_mask
                                    end_mask = end_seg.cropped_mask

                                    start_resized = cv2.resize(
                                        start_mask,
                                        (crop_w, crop_h),
                                        interpolation=cv2.INTER_LINEAR,
                                    )
                                    end_resized = cv2.resize(
                                        end_mask,
                                        (crop_w, crop_h),
                                        interpolation=cv2.INTER_LINEAR,
                                    )

                                    new_mask = lerp(
                                        start_resized, end_resized, t
                                    )
                                else:
                                    new_mask = np.ones(
                                        (crop_h, crop_w), dtype=np.float32
                                    )

                                timeline[last_valid_idx + j] = core.SEG(
                                    None,
                                    new_mask,
                                    1.0,
                                    new_crop,
                                    new_bbox,
                                    label,
                                    None,
                                )
                        last_valid_idx = i

            last_valid_seg = None
            for i in range(total_frames):
                if timeline[i] is not None:
                    last_valid_seg = timeline[i]
                elif last_valid_seg is not None:
                    timeline[i] = last_valid_seg

            if last_valid_seg is not None:
                for i in range(total_frames - 1, -1, -1):
                    if timeline[i] is not None:
                        last_valid_seg = timeline[i]
                    elif last_valid_seg is not None:
                        timeline[i] = last_valid_seg

        final_segs_by_frame = [[] for _ in range(total_frames)]
        for label in sorted(list(all_labels_found)):
            timeline = structured_segs[label]
            for i in range(total_frames):
                if timeline[i] is not None:
                    final_segs_by_frame[i].append(timeline[i])

        original_dims = (image.shape[1], image.shape[2])
        final_segs_object = (original_dims, final_segs_by_frame)
        logging.info(
            f"[FaceMesh Batch & Fill] Completed. Processed {len(final_segs_by_frame)} frames."
        )
        return (final_segs_object,)


class MTB_SegsToCombinedMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, segs):
        import impact.core as core
        import impact.utils as utils

        outputs = []
        if isinstance(segs[1], list) and len(segs[1]):
            if isinstance(segs[1][0], list):
                for seg in segs[1]:
                    mask = core.segs_to_combined_mask((segs[0], seg))
                    mask = utils.make_3d_mask(mask)
                    outputs.append(mask)

                return (
                    torch.stack(outputs, dim=0)
                    .permute(0, 2, 3, 1)
                    .squeeze(-1),
                )

            else:
                mask = core.segs_to_combined_mask(segs)
                mask = utils.make_3d_mask(mask)

                return (mask,)
        return (torch.zeros((0, 10, 10, 1)),)


__nodes__ = [MTB_FaceMeshBatchToSEGSAndFill, MTB_SegsToCombinedMaskBatch]
