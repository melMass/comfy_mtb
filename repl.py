import base64
import code
import io
import re
import sys
from contextlib import redirect_stderr, redirect_stdout

# import matplotlib.pyplot as plt
import numpy as np
import torch
from aiohttp import web
from PIL import Image
from rich.console import Console
from rich.traceback import Traceback

from .log import log

try:
    import pyflakes.api
    import pyflakes.reporter

    _HAS_LINT = True
except ImportError:
    print(
        "ComfyREPL: pyflakes not found. Linting will be disabled. Install with 'pip install pyflakes'."
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
#         "ComfyREPL: ruff not found. Linting will be disabled. Install with 'pip install ruff'."
#     )
#     _HAS_LINT = False

# --- Audio/Video Libraries ---
try:
    import scipy.io.wavfile

    _HAS_SCIPY = True
except ImportError:
    print(
        "ComfyREPL: SciPy not found. Audio display will be disabled. Install with 'pip install scipy'."
    )
    _HAS_SCIPY = False

try:
    import imageio
    import imageio.plugins.ffmpeg  # Ensure ffmpeg plugin is available

    _HAS_IMAGEIO = True
except ImportError:
    print(
        "ComfyREPL: Imageio or imageio-ffmpeg not found. Video display will be disabled. Install with 'pip install imageio imageio-ffmpeg'."
    )
    _HAS_IMAGEIO = False


# --- Audio Display ---
class AudioDisplay:
    def __init__(self, samples, sample_rate):
        if not _HAS_SCIPY:
            raise ImportError("Audio display requires scipy and numpy.")
        if not isinstance(samples, (np.ndarray, torch.Tensor)):
            raise TypeError(
                "Audio samples must be a numpy array or torch tensor."
            )
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()

        # Ensure samples are in a format scipy.io.wavfile can handle (e.g., int16, float32)
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
        return f'<audio controls src="data:audio/wav;base64,{base64_data}" style="margin: 5px 0;"/>'


def render_audio(samples, sample_rate):
    """
    Render audio samples as an HTML audio player.

    Args:
        samples (np.ndarray or torch.Tensor): Audio samples.
        sample_rate (int): Sample rate in Hz.

    Returns
    -------
        AudioDisplay: An object that will render as an HTML audio player.
    """
    return AudioDisplay(samples, sample_rate)


# --- Display Classes ---
class VideoDisplay:
    def __init__(self, frames, fps=24, options=None):
        if not _HAS_IMAGEIO:  # numpy/PIL/torch needed for frames
            raise ImportError(
                "Video display requires imageio, imageio-ffmpeg, and image libraries (numpy, Pillow, torch)."
            )

        self.frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                self.frames.append(np.array(frame))
            elif isinstance(frame, np.ndarray):
                # Ensure HWC and uint8
                if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:  # CHW
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
                if np_frame.ndim == 3 and np_frame.shape[0] in [
                    1,
                    3,
                    4,
                ]:  # CHW
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
                    f"Unsupported frame type: {type(frame)}. Must be PIL.Image, numpy.ndarray, or torch.Tensor."
                )

        self.fps = fps
        self.options = options if options is not None else {}

    def _to_mp4_base64(self):
        buffer = io.BytesIO()
        try:
            # Use imageio to write frames to an in-memory MP4 file
            imageio.mimwrite(
                buffer,
                self.frames,
                format="mp4",
                fps=self.fps,
                codec="libx264",
                quality=8,
            )  # quality 1-10
            video_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return video_base64
        except Exception as e:
            return f"<div style='color: red;'>Error encoding video: {e}</div>"

    def _repr_html_(self):
        base64_data = self._to_mp4_base64()
        if base64_data.startswith("<div"):  # Check if it's an error message
            return base64_data

        # Build HTML options string
        option_str = ""
        for key, value in self.options.items():
            if isinstance(value, bool) and value:
                option_str += f" {key}"
            elif isinstance(value, str):
                option_str += f' {key}="{value}"'
            else:
                option_str += f' {key}="{value}"'  # Fallback for numbers etc.

        return f'<video controls src="data:video/mp4;base64,{base64_data}" style="max-width: 100%; height: auto; border: 1px solid #555; margin: 5px 0;"{option_str}/>'


def render_video(batch_tensor_or_array_of_pil_images, fps=24, options=None):
    """
    Render video frames as an HTML video player.

    Args:
        batch_tensor_or_array_of_pil_images (list of PIL.Image, np.ndarray, or torch.Tensor):
            A list of frames, or a single batch tensor/array (B, H, W, C) or (B, C, H, W).
        fps (int): Frames per second.
        options (dict): Dictionary of HTML <video> tag attributes (e.g., {"loop": True, "autoplay": True}).

    Returns
    -------
        VideoDisplay: An object that will render as an HTML video player.
    """
    frames_list = []
    if isinstance(
        batch_tensor_or_array_of_pil_images, (np.ndarray, torch.Tensor)
    ):
        # Assume it's a batch tensor/array
        for i in range(batch_tensor_or_array_of_pil_images.shape[0]):
            frames_list.append(batch_tensor_or_array_of_pil_images[i])
    elif isinstance(batch_tensor_or_array_of_pil_images, list):
        frames_list = batch_tensor_or_array_of_pil_images
    else:
        raise TypeError(
            "Input for render_video must be a list of frames or a batch tensor/array."
        )

    return VideoDisplay(frames_list, fps, options)


class ComfyREPLBackend:
    def __init__(self):
        self.repl_consoles: dict[str, code.InteractiveConsole] = {}
        # self.repl_console = None
        self.image_outputs = []
        self.audio_outputs = []
        self.video_outputs = []
        self._original_displayhook = sys.displayhook
        # self._init_repl_console()

    @staticmethod
    def _init_repl_console():
        """Define the globals that will be available in the REPL session."""
        repl_globals = {"__builtins__": __builtins__}
        # repl_globals["plt"] = plt
        repl_globals["np"] = np
        repl_globals["Image"] = Image
        repl_globals["torch"] = torch

        repl_globals["repl_display"] = _repl_display_image
        if _HAS_SCIPY:
            repl_globals["render_audio"] = render_audio
        if _HAS_IMAGEIO:
            repl_globals["render_video"] = render_video

        return code.InteractiveConsole(locals=repl_globals)

    def _custom_displayhook(self, value):
        """
        Displayhook that capture and process image, audio, video objects.

        For other objects, it fallsback to the original displayhook.
        """
        if value is None:
            return

        # Attempt to handle as an image
        if (
            isinstance(value, (Image.Image, np.ndarray, torch.Tensor))
            # or (
            #     hasattr(value, "figure")
            #     and isinstance(value.figure, plt.Figure)
            # )
            # or isinstance(value, plt.Figure)
        ):
            img_html = _repl_display_image(value)
            self.image_outputs.append(img_html)
            return

        # Attempt to handle as audio
        elif isinstance(value, AudioDisplay):
            audio_html = value._repr_html_()
            self.audio_outputs.append(audio_html)
            return

        # Attempt to handle as video
        elif isinstance(value, VideoDisplay):
            video_html = value._repr_html_()
            self.video_outputs.append(video_html)
            return

        else:
            # If not a special media type, let the original displayhook handle it.
            self._original_displayhook(value)

    def _console_to_html(
        self, stream: io.StringIO | Traceback, width: int = 120
    ) -> str:
        if isinstance(stream, io.StringIO):
            captured_text_output = stream.getvalue()
        else:
            captured_text_output = Traceback

        html_console = Console(
            file=io.StringIO(), record=True, force_terminal=True, width=width
        )
        html_console.print(captured_text_output)

        return html_console.export_html(inline_styles=True)

    def get_console(self, node_name: str):
        console = self.repl_consoles.get(node_name)
        if console:
            return console

        console = self._init_repl_console()
        self.repl_consoles[node_name] = console
        return self.repl_consoles[node_name]

    def execute_code(self, node_name: str, code: str):
        # Clear outputs from previous execution
        self.image_outputs = []
        self.audio_outputs = []
        self.video_outputs = []

        output_html = ""
        error_message = None

        repl_console = self.get_console(node_name)

        string_io = io.StringIO()

        # Temporarily patch sys.displayhook
        sys.displayhook = self._custom_displayhook

        try:
            with redirect_stdout(string_io), redirect_stderr(string_io):
                for line in code.splitlines():
                    repl_console.push(line)

            full_rich_html = self._console_to_html(string_io)
            match = re.search(
                r"<body.*?>(.*?)</body>", full_rich_html, re.DOTALL
            )
            if match:
                output_html = match.group(1)
            else:
                output_html = full_rich_html

            # Append any captured media HTML *after* the rich text output
            for img_html in self.image_outputs:
                output_html += img_html
            for audio_html in self.audio_outputs:
                output_html += audio_html
            for video_html in self.video_outputs:
                output_html += video_html

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
        finally:
            sys.displayhook = (
                self._original_displayhook
            )  # Always restore original displayhook

        return {"output_html": output_html, "error": error_message}

    def lint_code(self, node_name: str, code: str):
        diagnostics = []

        if not _HAS_LINT:
            diagnostics.append(
                {
                    "row": 0,
                    "column": 0,
                    "text": "Pyflakes not installed. Linting disabled. Install with 'uv add pyflakes'.",
                    "type": "warning",
                }
            )
            return web.json_response({"diagnostics": diagnostics})

        # Use a custom reporter to capture messages
        class PyflakesReporter(pyflakes.reporter.Reporter):
            def __init__(self):
                self.messages = []
                # Suppress stdout/stderr from pyflakes itself
                self._stdout = io.StringIO()
                self._stderr = io.StringIO()
                super().__init__(self._stdout, self._stderr)

            def flake(self, message):
                # Ace editor expects 0-indexed row, pyflakes gives 1-indexed lineno
                self.messages.append(
                    {
                        "row": message.lineno - 1,
                        "column": message.col,
                        "text": str(message),
                        "type": "warning",  # pyflakes usually gives warnings
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
        pyflakes.api.check(code, node_name, reporter)

        return {"diagnostics": reporter.messages}

    def lint_code_ruff(self, code: str):
        diagnostics = []

        if not _HAS_LINT:
            diagnostics.append(
                {
                    "row": 0,
                    "column": 0,
                    "text": "Ruff not installed. Linting disabled. Install with 'pip install ruff'.",
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
            # "plt",
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
                # Ruff's error codes: F (Pyflakes), E (Pycodestyle), W (Pycodestyle warning), I (isort), N (naming), etc.
                # F821: Undefined name (often an error)
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


def _repl_display_image(img_data):
    """
    Internal function to convert image data (PIL, numpy, torch, matplotlib) to base64 HTML.
    """
    pil_img = None
    # fig = None

    if isinstance(img_data, Image.Image):
        pil_img = img_data
    elif isinstance(img_data, np.ndarray):
        # Handle different numpy array shapes (HWC, CHW)
        if img_data.ndim == 3:
            if img_data.shape[0] in [1, 3, 4]:  # Likely CHW
                if img_data.shape[0] == 1:  # Grayscale
                    img_data = img_data.squeeze(0)
                else:  # Color
                    img_data = np.transpose(img_data, (1, 2, 0))  # CHW to HWC
            # Ensure it's uint8 for PIL, assuming float [0,1] or int [0,255]
            if img_data.dtype != np.uint8:
                img_data = (
                    (img_data * 255).astype(np.uint8)
                    if img_data.max() <= 1.0
                    else img_data.astype(np.uint8)
                )
        pil_img = Image.fromarray(img_data)
    elif isinstance(img_data, torch.Tensor):
        # Move to CPU, convert to numpy
        np_img = img_data.detach().cpu().numpy()
        # Handle different tensor shapes (CHW, HWC)
        if np_img.ndim == 3:
            if np_img.shape[0] in [1, 3, 4]:  # Likely CHW
                if np_img.shape[0] == 1:  # Grayscale
                    np_img = np_img.squeeze(0)
                else:  # Color
                    np_img = np.transpose(np_img, (1, 2, 0))  # CHW to HWC
            # Ensure it's uint8 for PIL, assuming float [0,1] or int [0,255]
            if np_img.dtype != np.uint8:
                np_img = (
                    (np_img * 255).astype(np.uint8)
                    if np_img.max() <= 1.0
                    else np_img.astype(np.uint8)
                )
        pil_img = Image.fromarray(np_img)
    # elif hasattr(img_data, "figure") and isinstance(
    #     img_data.figure, plt.Figure
    # ):
    #     # If it's a matplotlib Axes object, get its figure
    #     fig = img_data.figure
    # elif isinstance(img_data, plt.Figure):
    #     fig = img_data
    else:
        return f"<div style='color: red;'>Unsupported image type for display: {type(img_data)}</div>"

    buffer = io.BytesIO()
    try:
        if pil_img:
            pil_img.save(buffer, format="PNG")
        # elif fig:
        #     fig.savefig(
        #         buffer, format="PNG", bbox_inches="tight", pad_inches=0.1
        #     )
        #     plt.close(
        #         fig
        #     )  # Close the figure to prevent it from showing up in other contexts
        else:
            return (
                "<div style='color: red;'>Could not process image data.</div>"
            )
    except Exception as e:
        return f"<div style='color: red;'>Error saving image: {e}</div>"

    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto; border: 1px solid #555; margin: 5px 0;"/>'


# Instantiate the backend class globally
_comfy_repl_backend = ComfyREPLBackend()


# Update aiohttp handlers to use the backend instance
async def repl_execute_code_handler(request):
    data = await request.json()

    name = data.get("name")

    if name is None:  # we send an error
        return web.Response(
            status=417, reason="Expectation Failed", text="Missing name key"
        )
    code = data.get("code", "")
    result = _comfy_repl_backend.execute_code(name, code)
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
    result = _comfy_repl_backend.lint_code(name, code)
    return web.json_response(result)


def setup_custom_web_routes(app: web.Application):
    """
    Function to register our custom web routes with the ComfyUI server.
    """
    log.info("ComfyREPL: Registering /mtb/execute route...")
    app.router.add_post("/mtb/execute", repl_execute_code_handler)
    app.router.add_post("/mtb/lint", repl_lint_code_handler)


# You can add more routes here if needed, e.g., for clearing state.
