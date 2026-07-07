import base64
import io
import textwrap
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

import torch
from rich import inspect
from rich.console import Console

from ..log import log
from ..utils import LazyProxyTensor, get_torch_tensor_info, tensor2pil

try:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("dark_background")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# region Decorator
def metadata(**meta_kwargs: Any) -> Callable[[Any], Any]:
    """Add metadata to method (`__meta__` dict)."""

    def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__meta__ = meta_kwargs
        return wrapper

    return decorator


# endregion
class UIResult(TypedDict):
    kind: Literal["text", "b64_images"]
    data: str


def indent_results(results: list[UIResult], by: str = "  "):
    for res in results:
        if res["kind"] == "text":
            log.debug(f"Indenting: {res['data']}")
            res["data"] = textwrap.indent(res["data"], by)

    return results


ProcessorResult = list[UIResult]


def _get_detailed_type_info(obj) -> str:
    type_info: list[str] = []

    type_name = type(obj).__name__
    type_info.append(f"Type: {type_name}")

    if isinstance(obj, torch.Tensor):
        return get_torch_tensor_info(obj)

    elif isinstance(obj, list | tuple):
        type_info.extend(
            [
                f"Length: {len(obj)}",
                f"Container type: {type_name}",
            ]
        )
        if obj:
            type_info.append(f"Element type: {type(obj[0]).__name__}")
    elif isinstance(obj, dict):
        type_info.extend(
            [
                f"Length: {len(obj)}",
                f"Keys: {list(obj.keys())}",
            ]
        )
    elif hasattr(obj, "__dict__"):
        attributes = [attr for attr in dir(obj) if not attr.startswith("_")]
        type_info.append(f"Attributes: {attributes}")

    return "\n".join(type_info)


def _apply_rich_results(processed, mode="none", title=""):
    processing_text = False
    acc = ""
    reshaped: list[UIResult] = []
    for i in range(len(processed)):
        if processed[i]["kind"] == "text":
            if not processing_text:
                processing_text = True
            acc += processed[i]["data"] + "\n"
            if len(processed) == (i + 1):
                reshaped.append(
                    UIResult(
                        kind="text", data=_apply_rich(acc, mode, title=title)
                    )
                )
        else:
            if processing_text:
                processing_text = False
                reshaped.append(
                    UIResult(
                        kind="text", data=_apply_rich(acc, mode, title=title)
                    )
                )
                acc = ""
            reshaped.append(processed[i])

    return reshaped
    # for item in processed:


# region processors
def _apply_rich(
    formatted: str | list[str], rich_mode: str | None = None, *, title=""
) -> str:
    if rich_mode is None:
        return (
            formatted if isinstance(formatted, str) else "\n".join(formatted)
        )

    from rich.console import Console

    console = Console(record=True)

    if isinstance(formatted, list):
        for line in formatted:
            console.print(line)
    else:
        console.print(formatted)

    CSV_CODE_FORMAT = """
<svg class="rich-terminal" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <!-- Generated with Rich https://www.textualize.io -->
    <style>

    @font-face {{
        font-family: "Fira Code";
        src: local("FiraCode-Regular"),
                url("https://cdnjs.cloudflare.com/ajax/libs/firacode/6.2.0/woff2/FiraCode-Regular.woff2") format("woff2"),
                url("https://cdnjs.cloudflare.com/ajax/libs/firacode/6.2.0/woff/FiraCode-Regular.woff") format("woff");
        font-style: normal;
        font-weight: 400;
    }}
    @font-face {{
        font-family: "Fira Code";
        src: local("FiraCode-Bold"),
                url("https://cdnjs.cloudflare.com/ajax/libs/firacode/6.2.0/woff2/FiraCode-Bold.woff2") format("woff2"),
                url("https://cdnjs.cloudflare.com/ajax/libs/firacode/6.2.0/woff/FiraCode-Bold.woff") format("woff");
        font-style: bold;
        font-weight: 700;
    }}

    .{unique_id}-matrix {{
        font-family: Fira Code, monospace;
        font-size: {char_height}px;
        line-height: {line_height}px;
        font-variant-east-asian: full-width;
    }}

    .{unique_id}-title {{
        font-size: 18px;
        font-weight: bold;
        font-family: arial;
    }}

    {styles}
    </style>

    <defs>
    <clipPath id="{unique_id}-clip-terminal">
      <rect x="0" y="0" width="{terminal_width}" height="{terminal_height}" />
    </clipPath>
    {lines}
    </defs>

    {chrome}
    <g  clip-path="url(#{unique_id}-clip-terminal)">
    {backgrounds}
    <g class="{unique_id}-matrix">
    {matrix}
    </g>
    </g>
</svg>
"""

    if rich_mode == "svg-window":
        return console.export_svg(title=title, code_format=CSV_CODE_FORMAT)
    elif rich_mode == "svg":
        return console.export_svg(
            title=title,
            code_format=CSV_CODE_FORMAT.replace("{chrome}", ""),
        )

    elif rich_mode == "html":
        CONSOLE_HTML_FORMAT = textwrap.dedent("""
        <div style="color:{foreground};">
            <code style="font-family:inherit">{code}</code>
        </div>
        """).strip()

        import rich.terminal_theme

        return console.export_html(
            inline_styles=True,
            code_format=CONSOLE_HTML_FORMAT,
            theme=rich.terminal_theme.MONOKAI,
        )

    log.error(f"Unknown rich mode: {rich_mode}")
    return formatted if isinstance(formatted, str) else "\n".join(formatted)


# endregion


# region conditions


# those are pretty dumb there is now probably a better way..
def is_condition(item):
    return (
        isinstance(item, list)
        and all(isinstance(i, list) for i in item)
        and isinstance(item[0][0], torch.Tensor)
    )


# endregion

RICH_MODE = Literal["none", "html", "svg", "svg-window"]


@runtime_checkable
class Processor(Protocol):
    """Generic protocol for processor functions."""

    def __call__(
        self, item: Any, *, as_type: bool = False, deep: bool = False
    ) -> ProcessorResult: ...


class MTB_Debug:
    """A debug node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"output_to_console": ("BOOLEAN", {"default": False})},
            "optional": {
                "as_detailed_types": ("BOOLEAN", {"default": False}),
                "deep_inspect": ("BOOLEAN", {"default": False}),
                "rich_mode": (
                    ("none", "html", "svg", "svg-window"),
                    {"default": "none"},
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "do_debug"
    CATEGORY = "mtb/debug"
    OUTPUT_NODE = True

    _processors: dict[type, Processor]

    def __init__(self):
        self._condition_processors = {is_condition: self._process_condition}
        self._class_name_processors = {
            "CLIP": self._process_clip,
            "VAE": self._process_vae,
        }
        self._processors = {
            torch.nn.Module: self._process_module,
            torch.Tensor: self._process_tensor,
            LazyProxyTensor: self._process_repr,
            list: self._process_container,
            tuple: self._process_container,
            dict: self._process_dict,
            bool: self._process_bool,
            str: self._process_primitive,
            int: self._process_primitive,
            float: self._process_primitive,
            type(None): self._process_primitive,
        }

    # - Dispatchers ------------------------------------------------------------
    def _dispatch_processor(
        self, item: Any, *, as_type=False, deep=False
    ) -> ProcessorResult:
        """Find and calls the appropriate processor for the given item."""
        # first conditions
        for c, process in self._condition_processors.items():
            if c(item):
                return process(item, as_type=as_type, deep=deep)

        # named class
        class_name = type(item).__name__
        if class_name in self._class_name_processors:
            return self._class_name_processors[class_name](
                item, as_type=as_type, deep=deep
            )

        # type based or unknown
        processor = self._processors.get(type(item), self._process_unknown)
        res = processor(item, as_type=as_type, deep=deep)

        return res

    def do_debug(
        self,
        **kwargs,
    ):
        output = {"ui": {"items": []}}

        settings = {k: kwargs.pop(k) for k in self.INPUT_TYPES()["optional"]}
        output_to_console = kwargs.pop("output_to_console")
        as_type = settings.get("as_detailed_types", False)
        deep = settings.get("deep_inspect", False)
        rich_mode = settings.get("rich_mode", "none")

        for input_name, item in kwargs.items():
            processed = self._dispatch_processor(
                item, as_type=as_type, deep=deep
            )
            if processed is None:
                continue

            if rich_mode != "none":
                title = f"{input_name} ({type(item).__name__})"
                processed = _apply_rich_results(processed, rich_mode, title)

            if output_to_console:
                log.info(f"- Input '{input_name}':")
                for p in processed:
                    if p["kind"] == "text":
                        log.info(f"  {p['data']}")
                    if p["kind"] == "b64_image":
                        log.info(f"  (contains {len(p['data'])} images)")

            output["ui"]["items"].append(
                {"input": input_name, "items": processed}
            )
        return output

    def _process_unknown(
        self, item: Any, *, as_type=False, deep=False
    ) -> ProcessorResult:
        console = Console(
            record=True,
            width=120,
        )

        console.print(f"Generic {type(item).__name__}", emoji=True)
        if as_type:
            inspect(item, console=console, all=deep, methods=deep, docs=deep)
        else:
            console.print(item, emoji=True)

        text_output = console.export_text(clear=True)

        return [UIResult(kind="text", data=text_output.strip())]

    def _process_repr(
        self, item: Any, as_type=False, deep=False
    ) -> ProcessorResult:
        return [{"kind": "text", "data": item.__repr__()}]

    def _process_primitive(
        self, item: Any, *, as_type=False, deep=False
    ) -> ProcessorResult:
        if as_type:
            return self._process_unknown(item, as_type=as_type, deep=deep)

        return [UIResult(kind="text", data=str(item))]

    def _process_bool(
        self, item: bool, *, as_type=False, deep=False
    ) -> ProcessorResult:  # noqa: FBT001
        return [{"kind": "text", "data": "True" if item else "False"}]

    def _process_clip(
        self, item: Any, *, as_type=False, deep=False
    ) -> ProcessorResult:
        try:
            clip_model = getattr(item, "cond_stage_model", None)
            tokenizer = getattr(item, "tokenizer", None)

            text = [UIResult(kind="text", data="CLIP")]
            if clip_model:
                text.append(UIResult(kind="text", data="CLIP Model:"))
                model_summary = self._process_module(
                    clip_model, as_type=as_type
                )
                if model_summary:
                    text.extend(indent_results(model_summary, "  "))
                else:
                    text.append(
                        UIResult(
                            kind="text",
                            data="[error] failed to get informations about clip model",
                        )
                    )

            if tokenizer:
                text.append(UIResult(kind="text", data="Tokenizer:"))
                vocab_size = getattr(tokenizer, "vocab_size", "N/A")
                text.append(
                    UIResult(
                        kind="text",
                        data=f"  Class: {type(tokenizer).__name__}\n  Vocab Size: {vocab_size}",
                    )
                )

            return text

        except Exception as e:
            log.error(f"Failed to process CLIP object: {e}")
            return self._process_unknown(item, as_type=as_type, deep=deep)

    def _process_condition(
        self, item: Any, *, as_type=False, deep=False
    ) -> ProcessorResult:
        count = len(item)
        result = [UIResult(kind="text", data=f"Conditions: {count}")]

        for cond in item:
            result.extend(self._preview_conditioning_tensor(cond[0]))

        return result

    def _process_vae(
        self, item: Any, *, as_type=False, deep=False
    ) -> ProcessorResult:
        try:
            vae_model = getattr(
                item, "first_stage_model", getattr(item, "vae", item)
            )
            text = [
                UIResult(kind="text", data="VAE"),
                UIResult(kind="text", data="Internal Model:"),
            ]

            model_summary = self._process_module(
                vae_model, as_type=as_type, deep=deep
            )
            text.extend(indent_results(model_summary, "  "))

            return text
        except Exception as e:
            log.error(f"Failed to process VAE object: {e}")
            return self._process_unknown(item, as_type=as_type, deep=deep)

    def _process_module(
        self, item: torch.nn.Module, *, as_type=False, deep=False
    ) -> ProcessorResult:
        if as_type and deep:
            return self._process_unknown(item, as_type=as_type, deep=deep)

        total_params = sum(p.numel() for p in item.parameters())
        trainable_params = sum(
            p.numel() for p in item.parameters() if p.requires_grad
        )
        try:
            device = next(item.parameters()).device
        except StopIteration:
            device = "cpu (no parameters)"

        train_percent = (
            f"{trainable_params / total_params:.2%}"
            if total_params > 0
            else "0.00%"
        )

        text = [
            f"Model: {type(item).__name__} on {device}",
            textwrap.dedent(f"""
            - Parameters: {total_params:,}
            - Trainable: {trainable_params:,} ({train_percent})
        """).strip(),
        ]
        return [{"kind": "text", "data": d} for d in text]

    def _process_tensor(
        self, item: torch.Tensor, *, as_type=False, deep=False
    ) -> ProcessorResult:
        is_latent = item.ndim == 4 and item.shape[1] == 4
        is_image = (
            not is_latent and item.ndim == 4 and item.shape[3] in [1, 3, 4]
        )
        is_conditioning = item.ndim == 3 and item.shape[2] in [
            768,
            1024,
            1152,
            1280,
            2048,
            4096,
        ]
        is_mask = (item.ndim == 2) or (item.ndim == 3 and not is_conditioning)

        if as_type:
            type_name = "Unknown Tensor"
            if is_latent:
                type_name = "Latent Tensor"
            elif is_image:
                type_name = "Image Tensor"
            elif is_conditioning:
                type_name = "CLIP Conditioning Tensor"
            elif is_mask:
                type_name = "Mask Tensor"
            return [
                {
                    "kind": "text",
                    "data": get_torch_tensor_info(item, name=type_name),
                }
            ]

        if is_image or is_mask:
            return self._render_image_tensor(item)
        if is_latent:
            return self._preview_latent_tensor(item)
        if is_conditioning:
            return self._preview_conditioning_tensor(item)
        return self._process_unknown(item, as_type=as_type, deep=deep)

    def _visualize_tensor_heatmap(
        self, tensor_2d: torch.Tensor, title: str
    ) -> str | None:
        if not MATPLOTLIB_AVAILABLE:
            log.warning("Matplotlib not found. Skipping tensor visualization.")
            return None
        if tensor_2d.ndim != 2:
            log.warning(
                f"Cannot visualize tensor with {tensor_2d.ndim} dimensions. Requires 2."
            )
            return None

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        im = ax.imshow(tensor_2d.cpu().numpy(), cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode(
            "utf-8"
        )

    def _render_image_tensor(self, item: torch.Tensor) -> ProcessorResult:
        is_mask = (item.ndim == 2) or (item.ndim == 3 and item.shape[-1] != 3)
        img_tensor = (
            item.unsqueeze(0) if item.ndim == 3 and not is_mask else item
        )
        img_tensor = item.unsqueeze(0) if item.ndim == 2 else img_tensor

        images = tensor2pil(img_tensor)
        b64_imgs = []
        for im in images:
            if is_mask:
                im = im.convert("L")
            buffered = io.BytesIO()
            im.save(buffered, format="PNG")
            b64_imgs.append(
                "data:image/png;base64,"
                + base64.b64encode(buffered.getvalue()).decode("utf-8")
            )
        return [UIResult(kind="b64_images", data=b64_imgs)]

    def _preview_latent_tensor(self, item: torch.Tensor) -> ProcessorResult:
        is_empty = "(empty)" if torch.count_nonzero(item) == 0 else ""
        stats = [
            f"Min: {item.min():.4f}",
            f"Max: {item.max():.4f}",
            f"Mean: {item.mean():.4f}",
        ]
        text = [
            get_torch_tensor_info(item, name="Latent Tensor"),
            is_empty,
        ] + stats

        result = [UIResult(kind="text", data=t) for t in text]
        vis_tensor = item[0].mean(dim=0)
        heatmap_b64 = self._visualize_tensor_heatmap(
            vis_tensor, "Latent Energy (Channel Mean)"
        )
        if heatmap_b64:
            result.append(UIResult(kind="b64_images", data=[heatmap_b64]))
        return result

    def _preview_conditioning_tensor(
        self, item: torch.Tensor
    ) -> ProcessorResult:
        _batch, tokens, embed_dim = item.shape
        text = [
            get_torch_tensor_info(item, name="CLIP Conditioning Tensor"),
            f"Token Count: {tokens}",
            f"Embedding Dim: {embed_dim}",
        ]

        result = [UIResult(kind="text", data=d) for d in text]
        heatmap_b64 = self._visualize_tensor_heatmap(
            item[0], "Token Embeddings (approx)"
        )
        if heatmap_b64:
            result.append(UIResult(kind="b64_images", data=[heatmap_b64]))
        return result

    def _process_container(
        self, item: list | tuple, *, as_type=False, deep=False
    ) -> ProcessorResult:
        if not item:
            return [UIResult(kind="text", data=f"Empty {type(item).__name__}")]

        container_type = type(item).__name__
        element_type = type(item[0]).__name__

        all_match = all(type(i) is type(item[0]) for i in item)

        result = [
            UIResult(
                kind="text",
                data=f"{container_type} of {len(item)} x {element_type}",
            ),
            UIResult(kind="text", data=f"(mixed types: {not all_match})"),
        ]

        if not as_type or (as_type and deep):
            for i, sub_item in enumerate(item):
                res = self._dispatch_processor(
                    sub_item, as_type=as_type, deep=deep
                )
                if res:
                    text = res[0].get("data", "Unknown")
                    res[0]["data"] = f"[{i}]: {text}"

                result.extend(res)

            return result

        first_item_result = self._dispatch_processor(
            item[0], as_type=as_type, deep=deep
        )
        if not first_item_result:
            return result

        return (
            result
            + [UIResult(kind="text", data="Preview of first element:")]
            + indent_results(first_item_result, "  - ")
        )

    def _process_dict(
        self, item: dict, *, as_type=False, deep=False
    ) -> ProcessorResult:
        if "pooled_output" in item and isinstance(
            item["pooled_output"], torch.Tensor
        ):
            return self._dispatch_processor(
                item["pooled_output"], as_type=as_type, deep=deep
            )

        if "samples" in item and isinstance(item.get("samples"), torch.Tensor):
            return self._dispatch_processor(
                item["samples"], as_type=as_type, deep=deep
            )

        if "waveform" in item and isinstance(
            item.get("waveform"), torch.Tensor
        ):
            waveform = item["waveform"]
            is_empty = "(empty) " if torch.count_nonzero(waveform) == 0 else ""
            text = textwrap.dedent(f"""
                Audio Waveform: {waveform.shape}{is_empty}
                Sample Rate: {item.get("sample_rate", "N/A")}
            """).strip()
            return [{"kind": "text", "data": text}]

        log.debug(
            f"Processing generic dict with rich inspector: {item.keys()}"
        )
        return self._process_unknown(item, as_type=as_type, deep=deep)


__nodes__ = [MTB_Debug]
