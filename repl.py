# ruff: noqa: S102 - Use of `exec` detected
# ruff: noqa: S307 - Use of possibly insecure function


# region imports
import ast
import base64
import code
import io
import re
import sys
import textwrap
import time
import urllib.parse
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any

import folder_paths
import numpy as np
import rich
import torch
from aiohttp import web
from PIL import Image
from rich.console import Console
from rich.traceback import Traceback

if TYPE_CHECKING:
    from .nodes.audio import AudioTensor

from .log import log
from .utils import singleton

# region safe imports
try:
    import pyflakes.api
    import pyflakes.reporter

    _HAS_LINT = True
except ImportError:
    log.error(
        textwrap.dedent("""
        ComfyREPL: pyflakes not found.
        Linting will be disabled.
        Install with 'pip install pyflakes'.
        """)
    )
    _HAS_LINT = False

# --- Linting Library ---
# try:
#     import ruff
#     import ruff.lint
#     import ruff.lint.linter
#     import ruff.settings
#
#     _HAS_LINT = True
# except ImportError:
#     print(
#         """ComfyREPL: ruff not found.
#            Linting will be disabled.
#            Install with 'pip install ruff'.
#         """
#     )
#     _HAS_LINT = False

# --- Audio/Video Libraries ---
try:
    import scipy.io.wavfile

    _HAS_SCIPY = True


except ImportError:
    log.warning(
        textwrap.dedent("""
        ComfyREPL: SciPy not found.
        Audio display will be disabled. Install with 'pip install scipy'.
        """)
    )
    _HAS_SCIPY = False

try:
    import imageio
    # import imageio.plugins.ffmpeg

    _HAS_IMAGEIO = True
except ImportError:
    log.warning(
        textwrap.dedent(
            """ComfyREPL: Imageio or imageio-ffmpeg not found.
            Video display will be disabled.
            Install with 'pip install imageio imageio-ffmpeg'
            or even better use uv.
        """
        )
    )
    _HAS_IMAGEIO = False

# endregion
# endregion


# region constants
# NOTE: The best way I can think of to make outputs dynamic
# is to register a large number from the python side and
# control their display dynamically from js (based on the dynamic inputs)
# but this is hacky and I would rather wait for V3 to start doing that.
SOCKET_COUNT = 5
# endregion


# region media_handlers
class _AudioDisplay:
    def __init__(self, samples, sample_rate):
        if not _HAS_SCIPY:
            raise ImportError("Audio display requires scipy and numpy.")
        if not isinstance(samples, np.ndarray | torch.Tensor):
            raise TypeError(
                "Audio samples must be a numpy array or torch tensor."
            )
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()

        # Ensure samples are in a format scipy.io.wavfile
        # can handle (e.g., int16, float32)
        if samples.dtype == np.float64:
            samples = samples.astype(np.float32)
        elif samples.dtype == np.int64:
            # Or scale to int32 if range requires
            samples = samples.astype(np.int16)

        self.samples = samples
        self.sample_rate = sample_rate

    def _to_wav_base64(self):
        buffer = io.BytesIO()
        try:
            scipy.io.wavfile.write(buffer, self.sample_rate, self.samples)
            audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return audio_base64
        except Exception as e:
            return f"<div style='color: red;'>Error encoding audio: {e}</div>"

    def _repr_html_(self):
        base64_data = self._to_wav_base64()
        if base64_data.startswith("<div"):
            return base64_data
        return f"""
        <audio
            controls
            src="data:audio/wav;base64,{base64_data}"
            style="margin: 5px 0;"
        />"""


def render_audio(samples: np.ndarray | torch.Tensor, sample_rate: int):
    """
    Render audio samples as an HTML audio player.

    Args:
        samples: Audio samples.
        sample_rate: Sample rate in Hz.

    Returns
    -------
        AudioDisplay: An object that will render as an HTML audio player.
    """
    return _AudioDisplay(samples, sample_rate)


class _VideoDisplay:
    def __init__(
        self, frames, fps=24, options=None, audio: "AudioTensor | None" = None
    ):
        if not _HAS_IMAGEIO:
            raise ImportError(
                textwrap.dedent(
                    """Video display requires
                    imageio, imageio-ffmpeg, and image libraries
                    (numpy, Pillow, torch).
                    """
                )
            )

        self.frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                self.frames.append(np.array(frame))
            elif isinstance(frame, np.ndarray):
                # ensure HWC and uint8
                # CHW
                if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
                    frame = np.transpose(frame, (1, 2, 0))
                if frame.dtype != np.uint8:
                    frame = (
                        (frame * 255).astype(np.uint8)
                        if frame.max() <= 1.0
                        else frame.astype(np.uint8)
                    )
                self.frames.append(frame)
            elif isinstance(frame, torch.Tensor):
                np_frame = frame.detach().cpu().numpy()
                # CHW
                if np_frame.ndim == 3 and np_frame.shape[0] in [
                    1,
                    3,
                    4,
                ]:
                    np_frame = np.transpose(np_frame, (1, 2, 0))
                if np_frame.dtype != np.uint8:
                    np_frame = (
                        (np_frame * 255).astype(np.uint8)
                        if np_frame.max() <= 1.0
                        else np_frame.astype(np.uint8)
                    )
                self.frames.append(np_frame)
            else:
                raise TypeError(
                    textwrap.dedent(f"""
                    Unsupported frame type: {type(frame)}.
                    Must be PIL.Image, numpy.ndarray, or torch.Tensor.
                """)
                )

        self.fps = fps
        self.options = options if options is not None else {}

        self.audio_array = None
        self.audio_samplerate = None
        if audio:
            if (
                isinstance(audio, dict)
                and "waveform" in audio
                and "sample_rate" in audio
            ):
                waveform_tensor = audio["waveform"]
                self.audio_samplerate = audio["sample_rate"]
                audio_np = waveform_tensor.detach().cpu().numpy()
                if audio_np.ndim == 1:
                    audio_np = audio_np[:, np.newaxis]
                elif (
                    audio_np.ndim > 1 and audio_np.shape[0] < audio_np.shape[1]
                ):
                    audio_np = np.transpose(audio_np)
                self.audio_array = audio_np
            else:
                log.warning("Audio provided in an incorrect format, ignoring.")

    def _save_to_disk_and_get_url(self):
        """Saves the video to a temp file and returns its URL."""
        output_dir = Path(folder_paths.get_temp_directory(), "mtb_repl_videos")
        output_dir.mkdir(exist_ok=True)

        filename = f"vid_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4()}.mp4"
        filepath = output_dir / filename

        try:
            imageio.mimwrite(
                uri=filepath,
                ims=self.frames,
                format="mp4",
                fps=self.fps,
                codec="libx264",
                quality=8,
                audio_array=self.audio_array,
                audio_samplerate=self.audio_samplerate,
                audio_codec="aac",
            )

            params = urllib.parse.urlencode(
                {
                    "filename": filename,
                    "subfolder": "mtb_repl_videos",
                    "type": "temp",
                }
            )

            return f"/view?{params}"

        except Exception as e:
            return f"<div style='color: red;'>Error saving video to disk: {e}</div>"

    def _to_mp4_base64(self):
        """[DEPRECATED] this bloats the client"""
        buffer = io.BytesIO()
        try:
            imageio.mimwrite(
                uri=buffer,
                ims=self.frames,
                format="mp4",
                fps=self.fps,
                codec="libx264",
                quality=8,
            )
            video_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return video_base64
        except Exception as e:
            return f"<div style='color: red;'>Error encoding video: {e}</div>"

    def _repr_html_(self, *, old=False):
        if old:
            base64_data = self._to_mp4_base64()
            if base64_data.startswith("<div"):
                return base64_data

            option_str = ""
            for key, value in self.options.items():
                if isinstance(value, bool) and value:
                    option_str += f" {key}"
                elif isinstance(value, str):
                    option_str += f' {key}="{value}"'
                else:
                    option_str += f' {key}="{value}"'

            return f"""
            <video
                controls
                src="data:video/mp4;base64,{base64_data}"
                style="max-width: 100%; height: auto; border: 1px solid #555; margin: 5px 0;"{option_str}
            />"""

        video_url = self._save_to_disk_and_get_url()

        if video_url.startswith("<div"):
            return video_url

        option_str = ""
        for key, value in self.options.items():
            if isinstance(value, bool) and value:
                option_str += f" {key}"
            else:
                option_str += f' {key}="{str(value)}"'

        return f"""
        <video 
            controls 
            src="{video_url}"
            style="max-width: 100%; height: auto; border: 1px solid #555; margin: 5px 0;"{option_str}>
        </video>
        """


def render_video(
    frames: torch.Tensor | list[np.ndarray] | list[Image.Image] | Any,
    fps=24,
    options=None,
    *,
    audio: "AudioTensor | None" = None,
):
    """
    Render video frames as an HTML video player.

    Args:
        frames: A list of frames, or a single batch tensor/array
                (B, H, W, C) or (B, C, H, W).
        fps (int): Frames per second.
        options (dict): Dictionary of HTML <video> tag attributes
                        (e.g., {"loop": True, "autoplay": True}).

    Returns
    -------
        VideoDisplay: An object that will render as an HTML video player.
    """
    frames_list = []
    if isinstance(frames, np.ndarray | torch.Tensor):
        for i in range(frames.shape[0]):
            frames_list.append(frames[i])
    elif isinstance(frames, list):
        frames_list = frames
    else:
        raise TypeError(
            textwrap.dedent("""
            Wrong input passed to render_video.
            must be a list of frames or a batch tensor/array.
        """)
        )

    return _VideoDisplay(frames_list, fps, options, audio=audio)


class _ImageDisplay:
    """Convert and display image-like objects."""

    def __init__(self, img_data: Any):
        self.pil_img = None
        self.error = None

        if isinstance(img_data, Image.Image):
            self.pil_img = img_data
        elif isinstance(img_data, np.ndarray):
            try:
                if img_data.ndim == 3 and img_data.shape[0] in [
                    1,
                    3,
                    4,
                ]:  # CHW
                    img_data = np.transpose(img_data, (1, 2, 0))
                if img_data.dtype != np.uint8:
                    img_data = (
                        (img_data * 255).astype(np.uint8)
                        if img_data.max() <= 1.0
                        else img_data.astype(np.uint8)
                    )
                self.pil_img = Image.fromarray(img_data)
            except Exception as e:
                self.error = f"Error converting NumPy array to image: {e}"
        elif isinstance(img_data, torch.Tensor):
            try:
                np_img = img_data.detach().cpu().numpy()
                if np_img.ndim == 3 and np_img.shape[0] in [1, 3, 4]:  # CHW
                    np_img = np.transpose(np_img, (1, 2, 0))
                if np_img.dtype != np.uint8:
                    np_img = (
                        (np_img * 255).astype(np.uint8)
                        if np_img.max() <= 1.0
                        else np_img.astype(np.uint8)
                    )
                self.pil_img = Image.fromarray(np_img)
            except Exception as e:
                self.error = f"Error converting torch.Tensor to image: {e}"
        else:
            self.error = (
                f"Unsupported image type for display: {type(img_data)}"
            )

    def _repr_html_(self) -> str:
        """Convert the internal PIL image to a base64 HTML string."""
        if self.error:
            return f"<div style='color: red;'>{self.error}</div>"
        if not self.pil_img:
            return (
                "<div style='color: red;'>Could not process image data.</div>"
            )

        buffer = io.BytesIO()
        try:
            self.pil_img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto; border: 1px solid #555; margin: 5px 0;"/>'
        except Exception as e:
            return f"<div style='color: red;'>Error saving image: {e}</div>"


def render_image(img_data: Any):
    """
    Render image-like data (PIL, NumPy, Torch) as an HTML image.

    This is a user-facing function for explicit display calls.
    """
    return _ImageDisplay(img_data)


# endregion


class SafeInputList(list):
    """Wrapper of list for custom error message."""

    def __getitem__(self, key):
        """Get item with error rewired."""
        try:
            return super().__getitem__(key)
        except IndexError as e:
            raise IndexError(
                textwrap.dedent("""
                Input index out of range.
                In live mode, the 'inputs' list is empty,
                You can use the guard 'if IS_LIVE:'
                to fill them only for the live mode""")
            ) from e


class ComfyReplAPI:
    """'API' available from the REPL instances."""

    def __init__(self):
        self.__color = ""
        pass

    def set_color(self, color: str):
        self.__color = color

    # def get_input(self, at: int, mock: Any) -> Any:
    #     pass

    def get_gradient_2d(self, start, stop, width, height, is_horizontal):
        if is_horizontal:
            return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
            return np.tile(np.linspace(start, stop, height), (width, 1)).T

    def get_gradient_3d(
        self,
        width,
        height,
        start_list,
        stop_list,
        is_horizontal_list,
        *,
        as_image=False,
    ):
        result = np.zeros((height, width, len(start_list)), dtype=np.float32)

        for i, (start, stop, is_horizontal) in enumerate(
            zip(start_list, stop_list, is_horizontal_list, strict=False)
        ):
            result[:, :, i] = self.get_gradient_2d(
                start, stop, width, height, is_horizontal
            )
        if as_image:
            return Image.fromarray(np.uint8(result))
        else:
            return result


@singleton
class ComfyREPLBackend:
    """Singleton "backend" for the REPL editor."""

    def __init__(self):
        self._repl_consoles: dict[str, code.InteractiveConsole] = {}

        self.media_outputs = []

        # self._original_displayhook = sys.displayhook
        self.handlers = []
        self.register_handlers()

    def register_handlers(self):
        self.handlers.extend(
            [
                self._handle_video,
                self._handle_audio,
                self._handle_image,
            ]
        )

    def _handle_image(self, value):
        """Handle PIL Images, and Numpy/Torch tensors that represent images."""
        if isinstance(value, _ImageDisplay):
            return value._repr_html_()

        if isinstance(value, Image.Image | np.ndarray | torch.Tensor):
            return _ImageDisplay(value)._repr_html_()

        return None

    def _handle_audio(self, value):
        """Handle AudioDisplay objects."""
        if isinstance(value, _AudioDisplay):
            return value._repr_html_()
        return None

    def _handle_video(self, value):
        """Handle VideoDisplay objects."""
        if isinstance(value, _VideoDisplay):
            return value._repr_html_()

        if isinstance(value, torch.Tensor) and len(value.shape) == 4:
            return render_video(value)._repr_html_()

        if (
            isinstance(value, list)
            and len(value) > 0
            and isinstance(value, Image.Image | np.ndarray | torch.Tensor)
        ):
            return render_video(value)._repr_html_()
        return None

    def _init_repl_console(self, *inputs):
        """Define the globals that will be available in the REPL session."""
        repl_locals = {"__builtins__": __builtins__}
        # repl_globals["plt"] = plt
        repl_locals["np"] = np
        repl_locals["Image"] = Image
        repl_locals["torch"] = torch
        repl_api = ComfyReplAPI()
        repl_locals["repl"] = repl_api

        # "globals"
        repl_locals["IS_LIVE"] = True

        filename = Path(folder_paths.get_input_directory()) / "example.png"
        log.debug(
            f"Looking for example in {filename.as_posix()}, {filename.exists()}"
        )

        if filename.exists():
            img = Image.open(filename)
            repl_locals["EXAMPLE"] = torch.from_numpy(
                np.array(img).astype(np.float32) / 255.0
            ).unsqueeze(0)
        else:
            repl_locals["EXAMPLE"] = torch.from_numpy(
                repl_api.get_gradient_3d(
                    1024,
                    1024,
                    (0, 0, 192),
                    (255, 255, 64),
                    (True, False, False),
                )
            ).unsqueeze(0)

        repl_locals["inputs"] = [None] * SOCKET_COUNT
        repl_locals["outputs"] = [None] * SOCKET_COUNT

        if len(inputs):
            repl_locals["inputs"] = SafeInputList(inputs)

        # internals but exposed for debug
        if _HAS_SCIPY:
            repl_locals["render_audio"] = render_audio
        if _HAS_IMAGEIO:
            repl_locals["render_video"] = render_video

        repl_locals["render_image"] = render_image

        return code.InteractiveConsole(locals=repl_locals)

    def _custom_displayhook(self, value):
        """
        Displayhook that capture and process image, audio, video objects.

        It can also look for renderable items inside lists, tuples, and dicts.
        For other objects, it fallsback to the original displayhook.
        """
        log.debug("Custom displayhook called")
        if value is None:
            return

        # handle with handler
        if self._handle_item(value):
            return

        # containers
        items_to_process = []
        if isinstance(value, list | tuple):
            items_to_process.extend(value)
        elif isinstance(value, dict):
            items_to_process.extend(value.values())

        if items_to_process:
            was_anything_rendered = False
            for item in items_to_process:
                if self._handle_item(item):
                    was_anything_rendered = True
                    # Item handled, move to the next item in the container
                    break

            # If we found any media inside the container, we consider the
            # displayhook's job done. We don't also print the container itself.
            if was_anything_rendered:
                return

        log.debug("Falling back to original display hook")
        # self._original_displayhook(value)
        sys.displayhook(value)

    def _handle_item(self, item):
        # handle with handler
        for handler in self.handlers:
            html_output = handler(item)
            if html_output:
                log.debug(f"Object handled by {handler.__name__}")
                self.media_outputs.append(html_output)
                return True
        return False

    def _console_to_html(
        self, stream: io.StringIO | Traceback, width: int = 120
    ) -> str:
        if isinstance(stream, io.StringIO):
            captured_text_output = stream.getvalue()
        else:
            log.debug(
                f"Steam is not an io.StringIO but a {type(stream).__name__}"
            )
            captured_text_output = stream  # Traceback

        html_console = Console(
            file=io.StringIO(), record=True, force_terminal=True, width=width
        )
        html_console.print(captured_text_output)

        return html_console.export_html(inline_styles=True)

    def get_console(self, name: str | None = None, *, reset=False):
        """Get or create a new named console."""
        if name is None:
            log.error("Cannot get console without providing a name (uuid)")
            return

        console = self._repl_consoles.get(name)

        if console and not reset:
            return console

        if reset:
            if console is not None:
                del self._repl_consoles[name]
                console = None

            console = self._init_repl_console()
            self._repl_consoles[name] = console
            return self._repl_consoles[name]

        log.debug(
            textwrap.dedent(f"""
                Creating new console for node {name}
                (active: {len(self._repl_consoles.keys())})
                """)
        )
        console = self._init_repl_console()
        self._repl_consoles[name] = console
        return self._repl_consoles[name]

    def get_outputs(
        self,
        *,
        name: str | None = None,
        console: code.InteractiveConsole | None = None,
    ):
        if name is None and console is None:
            raise ValueError("Either id or console must be provided")

        if name is not None:
            console = self.get_console(name)

        if console is None:
            raise RuntimeError("No console found")

        if "outputs" in console.locals:
            outputs = console.locals["outputs"]
            # guards
            if outputs is None:
                outputs = [None] * SOCKET_COUNT
            elif isinstance(outputs, list) and len(outputs) > SOCKET_COUNT:
                raise RuntimeError(
                    textwrap.dedent(f"""
                    Too many outputs: {len(outputs)} vs {SOCKET_COUNT}
                    Outputs can be smaller then {SOCKET_COUNT} but not higher.
                """)
                )

            # enlist if needed
            if not isinstance(outputs, list):
                outputs = [outputs]

            # pad the outputs
            if len(outputs) < SOCKET_COUNT:
                outputs = outputs + [None] * (SOCKET_COUNT - len(outputs))
        else:
            outputs = [None] * SOCKET_COUNT

        return outputs

    def execute_code(
        self,
        code: str,
        *,
        name: str | None = None,
        console=None,
        reset=False,
        inputs: list[Any] | None = None,
    ):
        log.debug(f"Executing code for node {name}")

        inputs = inputs or []

        if not console and not name:
            raise ValueError("Either name or console must be provided")
        if console is not None and reset:
            raise ValueError(
                textwrap.dedent("""
                Cannot specify both console and reset,
                use name and reset instead
                """)
            )

        self.media_outputs = []
        output_html = ""
        error_message = None

        repl_console = console or self.get_console(name, reset=reset)
        if not repl_console:
            log.error(f"Failed to get console named {name}")
            return

        string_io = io.StringIO()
        repl_console.locals["inputs"] = SafeInputList(inputs)

        # Temporarily patch sys.displayhook
        # sys.displayhook = self._custom_displayhook

        try:
            log.debug(f"CODE SENT:\n\n{code}")
            with redirect_stdout(string_io), redirect_stderr(string_io):
                try:
                    tree = ast.parse(code)
                except SyntaxError as e:
                    raise e

                if tree.body:
                    # split if last statement is an expr
                    if isinstance(tree.body[-1], ast.Expr):
                        setup_module = ast.Module(
                            body=tree.body[:-1], type_ignores=[]
                        )
                        setup_code = compile(
                            setup_module, "<mtb-repl-setup>", "exec"
                        )

                        # the last expr
                        last_expr = ast.Expression(body=tree.body[-1].value)
                        last_code = compile(
                            last_expr, "<mtb-repl-eval>", "eval"
                        )

                        exec(setup_code, repl_console.locals)
                        result = eval(last_code, repl_console.locals)

                        # required as runsource doesn't trigger displayhook
                        self._custom_displayhook(result)

                        if "outputs" in repl_console.locals and isinstance(
                            repl_console.locals["outputs"], list | tuple
                        ):
                            log.warning(
                                textwrap.dedent("""
                                Outputs are defined
                                but the display hook will overwrite them""")
                            )
                            repl_console.locals["outputs"] = result

                    # no expression as last statement
                    else:
                        exec(code, repl_console.locals)

                # NOTE: old
                # repl_console.runsource(code, "mtb-repl", "exec")


            full_rich_html = self._console_to_html(string_io)

            match = re.search(
                r"<body.*?>(.*?)</body>", full_rich_html, re.DOTALL
            )
            output_html = match.group(1) if match else full_rich_html

            output_html += "".join(self.media_outputs)

        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            rich_traceback = Traceback.from_exception(
                exc_type,
                exc_value,
                exc_traceback,
                show_locals=True,
                suppress=[__file__],
            )
            # error_console = Console(
            # file=io.StringIO(), record=True, force_terminal=True, width=120
            # )
            # error_console.print(rich_traceback)
            # full_error_html = error_console.export_html(inline_styles=True)

            full_error_html = self._console_to_html(rich_traceback)

            match = re.search(
                r"<body.*?>(.*?)</body>", full_error_html, re.DOTALL
            )
            output_html = match.group(1) if match else full_error_html

            error_message = str(e)
        # finally:
        #     sys.displayhook = (
        #         self._original_displayhook
        #     )  # Always restore original displayhook

        return {"output_html": output_html, "error": error_message}

    @property
    def active_consoles(self):  # noqa: N805
        return [str(k) for k in self._repl_consoles]

    def lint_code(self, *, name: str = "", code: str):
        log.debug(f"Linting code for node {name}")
        diagnostics = []

        if not _HAS_LINT:
            diagnostics.append(
                {
                    "row": 0,
                    "column": 0,
                    "text": textwrap.dedent("""
                        Pyflakes not installed.
                        Linting disabled.
                        Install with 'uv add pyflakes'.
                    """),
                    "type": "warning",
                }
            )
            return web.json_response({"diagnostics": diagnostics})

        custom_globals = {"render_video", "render_audio", "repl_display"}

        # Use a custom reporter to capture messages
        class PyflakesReporter(pyflakes.reporter.Reporter):
            def __init__(self):
                self.messages = []
                # Suppress stdout/stderr from pyflakes itself
                self._stdout = io.StringIO()
                self._stderr = io.StringIO()
                super().__init__(self._stdout, self._stderr)

            def flake(self, message):
                # Ace editor expects 0-indexed row
                # pyflakes gives 1-indexed lineno

                import pyflakes.messages

                kind = "warning"

                if isinstance(message, pyflakes.messages.UndefinedName):
                    kind = "error"

                    if (
                        message.message_args
                        and message.message_args[0] in custom_globals
                    ):
                        return

                log.info("result from flake")
                console = rich.console.Console(stderr=True)
                rich.inspect(message, console=console)

                self.messages.append(
                    {
                        "row": message.lineno - 1,
                        "column": message.col,
                        "text": str(message) + "prout",
                        "type": kind,
                    }
                )

            def unexpectedError(self, filename, msg):
                self.messages.append(
                    {
                        "row": 0,
                        "column": 0,
                        "text": f"Pyflakes internal error: {msg}",
                        "type": "error",
                    }
                )

            def syntaxError(self, filename, msg, lineno, offset, text):
                log.info(f"Received {text} to syntax error")
                self.messages.append(
                    {
                        "row": lineno - 1,  # Ace is 0-indexed
                        "column": offset,
                        "text": f"Syntax Error: {msg}",
                        "type": "error",
                    }
                )

        reporter = PyflakesReporter()
        pyflakes.api.check(code, name, reporter)

        return {"diagnostics": reporter.messages}

    def lint_code_ruff(self, code: str):
        diagnostics = []

        if not _HAS_LINT:
            diagnostics.append(
                {
                    "row": 0,
                    "column": 0,
                    "text": textwrap.dedent("""
                        Ruff not installed.
                        Linting disabled.
                        Install with 'pip install ruff'.
                    """),
                    "type": "warning",
                }
            )
            return {"diagnostics": diagnostics}

        # Define the builtins/globals that Ruff should recognize
        # These are the names we inject into the REPL's scope
        repl_builtins = [
            "repl_display",
            "render_audio",
            "render_video",
            "plt",
            "np",
            "Image",
            "torch",
        ]

        try:
            # Lint the code using Ruff's programmatic API
            result = ruff.lint.linter.lint_stdin(
                code.encode("utf-8"),
                path="<stdin>",
                builtins=repl_builtins,
            )

            for diagnostic in result.diagnostics:
                diag_type = "warning"  # Default
                # Ruff's error codes:
                # F (Pyflakes)
                # E (Pycodestyle)
                # W (Pycodestyle warning)
                # I (isort)
                # N (naming)
                # ...
                # F821: Undefined name
                if (
                    diagnostic.kind.code.startswith("E")
                    or diagnostic.kind.code == "F821"
                ):
                    diag_type = "error"
                elif diagnostic.kind.code.startswith("W"):
                    diag_type = "warning"

                diagnostics.append(
                    {
                        "row": diagnostic.location.row - 1,  # Ace is 0-indexed
                        "column": diagnostic.location.column
                        - 1,  # Ace is 0-indexed
                        "text": diagnostic.message,
                        "type": diag_type,
                    }
                )

        except Exception as e:
            diagnostics.append(
                {
                    "row": 0,
                    "column": 0,
                    "text": f"Ruff internal error: {e}",
                    "type": "error",
                }
            )

        return {"diagnostics": diagnostics}


_comfy_repl_backend = ComfyREPLBackend()


# region endpoint handlers
async def repl_execute_code_handler(request):
    data = await request.json()

    name = data.get("name")
    reset = data.get("reset")

    if reset is None:
        reset = False

    if name is None:
        return web.Response(
            status=417, reason="Expectation Failed, missing name key"
        )

    code = data.get("code", "")
    result = _comfy_repl_backend.execute_code(code, name=name, reset=reset)

    return web.json_response(result)


async def repl_lint_code_handler(request):
    data = await request.json()
    name = data.get("name")

    if name is None:  # we send an error
        return web.Response(
            status=417, reason="Expectation Failed", text="Missing name key"
        )
        # raise web.HTTPExpectationFailed(
        # reason="Missing name key (reason)", text="Missing name key (text)"
        # )

    code = data.get("code", "")
    result = _comfy_repl_backend.lint_code(name=name, code=code)
    return web.json_response(result)


def setup_custom_web_routes(app: web.Application):
    """Register REPL routes."""
    log.info("ComfyREPL: Registering repl routes...")
    app.router.add_post("/mtb/execute", repl_execute_code_handler)
    app.router.add_post("/mtb/lint", repl_lint_code_handler)


# endregion
