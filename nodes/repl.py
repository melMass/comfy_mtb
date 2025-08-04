import textwrap

from comfy.comfy_types import IO

from ..repl import SOCKET_COUNT, ComfyREPLBackend
from ..utils import log


class MTB_Repl:
    """Write python code from within a python graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uuid": ("STRING",),
                "code": ("CODE_EDITOR", {"lang": "python"}),
            },
            "optional": {
                f"input_{i:02d}": (IO.ANY,) for i in range(1, SOCKET_COUNT + 1)
            }
            | {
                "reset": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "reset state at each run",
                        "label_off": "keep mutated state",
                    },
                ),
                "propagate": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "allow",
                        "label_off": "block forward",
                    },
                ),
            },
        }

    EXPERIMENTAL = True
    RETURN_TYPES = tuple(IO.ANY for _ in range(SOCKET_COUNT))
    RETURN_NAMES = tuple(f"output_{i:02d}" for i in range(1, SOCKET_COUNT + 1))
    FUNCTION = "run"
    CATEGORY = "mtb/repl"
    OUTPUT_NODE = True

    # @classmethod
    # def IS_CHANGED(cls, code: str, **kwargs):
    #     code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    #     return code_hash

    # TODO: hide uuid from the frontend
    # same for propagate, it should be managed from our run button in js
    def run(
        self, *, uuid: str, code: str, reset=False, propagate=True, **kwargs
    ):
        log.debug(f"Received code from frontend:\n{code}")
        log.debug(f"Kwargs: {kwargs}")

        repl_backend = ComfyREPLBackend()

        console = repl_backend.get_console(uuid, reset=reset)

        if not console:
            # TODO: handle this case automatically
            raise RuntimeError(
                textwrap.dedent(f"""
                No matching console found for {uuid}
                active console: {ComfyREPLBackend.active_consoles}
                """)
            )

        inputs = list(kwargs.values())

        console.locals["IS_LIVE"] = False

        ui = repl_backend.execute_code(
            code,
            console=console,
            inputs=inputs,
            # , reset=True
        )
        if ui is None:
            ui = {"output_html": [""], "error": [""]}
        else:
            ui = {"output_html": [ui["output_html"]], "error": [ui["error"]]}

        # process outputs
        outputs = repl_backend.get_outputs(console=console)

        log.debug(ui)
        # return outputs
        if propagate:
            results = tuple(outputs)
        else:
            from comfy_execution.graph import ExecutionBlocker

            results = [ExecutionBlocker(None)] * SOCKET_COUNT

        return {"ui": ui, "result": results}


__nodes__ = [MTB_Repl]
