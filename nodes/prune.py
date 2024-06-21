from pathlib import Path

import safetensors.torch
import torch
import tqdm

from ..log import log
from ..utils import Operation, Precision
from ..utils import output_dir as comfy_out_dir

PRUNE_DATA = {
    "known_junk_prefix": [
        "embedding_manager.embedder.",
        "lora_te_text_model",
        "control_model.",
    ],
    "nai_keys": {
        "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
        "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
        "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
    },
}

# position_ids in clip is int64. model_ema.num_updates is int32
dtypes_to_fp16 = {torch.float32, torch.float64, torch.bfloat16}
dtypes_to_bf16 = {torch.float32, torch.float64, torch.float16}
dtypes_to_fp8 = {torch.float32, torch.float64, torch.bfloat16, torch.float16}


class MTB_ModelPruner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "unet": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "required": {
                "save_separately": ("BOOLEAN", {"default": False}),
                "save_folder": ("STRING", {"default": "checkpoints/ComfyUI"}),
                "fix_clip": ("BOOLEAN", {"default": True}),
                "remove_junk": ("BOOLEAN", {"default": True}),
                "ema_mode": (
                    ("disabled", "remove_ema", "ema_only"),
                    {"default": "remove_ema"},
                ),
                "precision_unet": (
                    Precision.list_members(),
                    {"default": Precision.FULL.value},
                ),
                "operation_unet": (
                    Operation.list_members(),
                    {"default": Operation.CONVERT.value},
                ),
                "precision_clip": (
                    Precision.list_members(),
                    {"default": Precision.FULL.value},
                ),
                "operation_clip": (
                    Operation.list_members(),
                    {"default": Operation.CONVERT.value},
                ),
                "precision_vae": (
                    Precision.list_members(),
                    {"default": Precision.FULL.value},
                ),
                "operation_vae": (
                    Operation.list_members(),
                    {"default": Operation.CONVERT.value},
                ),
            },
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    CATEGORY = "mtb/prune"
    FUNCTION = "prune"

    def convert_precision(self, tensor: torch.Tensor, precision: Precision):
        precision = Precision.from_str(precision)
        log.debug(f"Converting to {precision}")
        match precision:
            case Precision.FP8:
                if tensor.dtype in dtypes_to_fp8:
                    return tensor.to(torch.float8_e4m3fn)
                log.error(f"Cannot convert {tensor.dtype} to fp8")
                return tensor
            case Precision.FP16:
                if tensor.dtype in dtypes_to_fp16:
                    return tensor.half()
                log.error(f"Cannot convert {tensor.dtype} to f16")
                return tensor
            case Precision.BF16:
                if tensor.dtype in dtypes_to_bf16:
                    return tensor.bfloat16()
                log.error(f"Cannot convert {tensor.dtype} to bf16")
                return tensor
            case Precision.FULL | Precision.FP32:
                return tensor

    def is_sdxl_model(self, clip: dict[str, torch.Tensor] | None):
        if clip:
            return (any(k.startswith("conditioner.embedders") for k in clip),)
        return False

    def has_ema(self, unet: dict[str, torch.Tensor]):
        return any(k.startswith("model_ema") for k in unet)

    def fix_clip(self, clip: dict[str, torch.Tensor] | None):
        if self.is_sdxl_model(clip):
            log.warn("[fix clip] SDXL not supported")
            return

        if clip is None:
            return

        position_id_key = (
            "cond_stage_model.transformer.text_model.embeddings.position_ids"
        )
        if position_id_key in clip:
            correct = torch.Tensor([list(range(77))]).to(torch.int64)
            now = clip[position_id_key].to(torch.int64)

            broken = correct.ne(now)
            broken = [i for i in range(77) if broken[0][i]]

            if len(broken) != 0:
                clip[position_id_key] = correct
                log.info(f"[Converter] Fixed broken clip\n{broken}")
            else:
                log.info(
                    "[Converter] Clip in this model is fine, skip fixing..."
                )

        else:
            log.info("[Converter] Missing position id in model, try fixing...")
            clip[position_id_key] = torch.Tensor([list(range(77))]).to(
                torch.int64
            )
        return clip

    def get_dicts(self, unet, clip, vae):
        clip_sd = clip.get_sd()
        state_dict = unet.model.state_dict_for_saving(
            clip_sd, vae.get_sd(), None
        )

        unet = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("model.diffusion_model")
        }
        clip = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("cond_stage_model")
            or k.startswith("conditioner.embedders")
        }
        vae = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("first_stage_model")
        }

        other = {
            k: v
            for k, v in state_dict.items()
            if k not in unet and k not in vae and k not in clip
        }

        return (unet, clip, vae, other)

    def do_remove_junk(self, tensors: dict[str, dict[str, torch.Tensor]]):
        need_delete: list[str] = []
        for layer in tensors:
            for key in layer:
                for jk in PRUNE_DATA["known_junk_prefix"]:
                    if key.startswith(jk):
                        need_delete.append(".".join([layer, key]))

        for k in need_delete:
            log.info(f"Removing junk data: {k}")
            del tensors[k]

        return tensors

    def prune(
        self,
        *,
        save_separately: bool,
        save_folder: str,
        fix_clip: bool,
        remove_junk: bool,
        ema_mode: str,
        precision_unet: Precision,
        precision_clip: Precision,
        precision_vae: Precision,
        operation_unet: str,
        operation_clip: str,
        operation_vae: str,
        unet: dict[str, torch.Tensor] | None = None,
        clip: dict[str, torch.Tensor] | None = None,
        vae: dict[str, torch.Tensor] | None = None,
    ):
        operation = {
            "unet": Operation.from_str(operation_unet),
            "clip": Operation.from_str(operation_clip),
            "vae": Operation.from_str(operation_vae),
        }
        precision = {
            "unet": Precision.from_str(precision_unet),
            "clip": Precision.from_str(precision_clip),
            "vae": Precision.from_str(precision_vae),
        }

        unet, clip, vae, _other = self.get_dicts(unet, clip, vae)

        out_dir = Path(save_folder)
        folder = out_dir.parent
        if not out_dir.is_absolute():
            folder = (comfy_out_dir / save_folder).parent

        if not folder.exists():
            if folder.parent.exists():
                folder.mkdir()
            else:
                raise FileNotFoundError(
                    f"Folder {folder.parent} does not exist"
                )

        name = out_dir.name
        save_name = f"{name}-{precision_unet}"
        if ema_mode != "disabled":
            save_name += f"-{ema_mode}"
        if fix_clip:
            save_name += "-clip-fix"

        if (
            any(o == Operation.CONVERT for o in operation.values())
            and any(p == Precision.FP8 for p in precision.values())
            and torch.__version__ < "2.1.0"
        ):
            raise NotImplementedError(
                "PyTorch 2.1.0 or newer is required for fp8 conversion"
            )

        if not self.is_sdxl_model(clip):
            for part in [unet, vae, clip]:
                if part:
                    nai_keys = PRUNE_DATA["nai_keys"]
                    for k in list(part.keys()):
                        for r in nai_keys:
                            if isinstance(k, str) and k.startswith(r):
                                new_key = k.replace(r, nai_keys[r])
                                part[new_key] = part[k]
                                del part[k]
                                log.info(
                                    f"[Converter] Fixed novelai error key {k}"
                                )
                                break

            if fix_clip:
                clip = self.fix_clip(clip)

        ok: dict[str, dict[str, torch.Tensor]] = {
            "unet": {},
            "clip": {},
            "vae": {},
        }

        def _hf(part: str, wk: str, t: torch.Tensor):
            if not isinstance(t, torch.Tensor):
                log.debug("Not a torch tensor, skipping key")
                return

            log.debug(f"Operation {operation[part]}")
            if operation[part] == Operation.CONVERT:
                ok[part][wk] = self.convert_precision(
                    t, precision[part]
                )  # conv_func(t)
            elif operation[part] == Operation.COPY:
                ok[part][wk] = t
            elif operation[part] == Operation.DELETE:
                return

        log.info("[Converter] Converting model...")

        for part_name, part in zip(
            ["unet", "vae", "clip", "other"],
            [unet, vae, clip],
            strict=False,
        ):
            if part:
                match ema_mode:
                    case "remove_ema":
                        for k, v in tqdm.tqdm(part.items()):
                            if "model_ema." not in k:
                                _hf(part_name, k, v)
                    case "ema_only":
                        if not self.has_ema(part):
                            log.warn("No EMA to extract")
                            return
                        for k in tqdm.tqdm(part):
                            ema_k = "___"
                            try:
                                ema_k = "model_ema." + k[6:].replace(".", "")
                            except Exception:
                                pass
                            if ema_k in part:
                                _hf(part_name, k, part[ema_k])
                            elif not k.startswith("model_ema.") or k in [
                                "model_ema.num_updates",
                                "model_ema.decay",
                            ]:
                                _hf(part_name, k, part[k])
                    case "disabled" | _:
                        for k, v in tqdm.tqdm(part.items()):
                            _hf(part_name, k, v)

                if save_separately:
                    if remove_junk:
                        ok = self.do_remove_junk(ok)

                    flat_ok = {
                        k: v
                        for _, subdict in ok.items()
                        for k, v in subdict.items()
                    }
                    save_path = (
                        folder / f"{part_name}-{save_name}.safetensors"
                    ).as_posix()
                    safetensors.torch.save_file(flat_ok, save_path)
                    ok: dict[str, dict[str, torch.Tensor]] = {
                        "unet": {},
                        "clip": {},
                        "vae": {},
                    }

        if save_separately:
            return ()

        if remove_junk:
            ok = self.do_remove_junk(ok)

        flat_ok = {
            k: v for _, subdict in ok.items() for k, v in subdict.items()
        }

        try:
            safetensors.torch.save_file(
                flat_ok, (folder / f"{save_name}.safetensors").as_posix()
            )
        except Exception as e:
            log.error(e)

        return ()


__nodes__ = [MTB_ModelPruner]
