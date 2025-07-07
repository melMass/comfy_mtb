import time
import uuid
from collections import OrderedDict
from typing import Any, TypedDict

from comfy.comfy_types.node_typing import IO as CIO
from server import PromptServer

from ..log import log


class Clock(TypedDict):
    name: str
    start: float
    end: float | None


active_timers: OrderedDict[str, Clock] = OrderedDict()

# TODO: lower this
MAX_CLOCKS = 50


class MTB_StartClock:
    """
    Starts a profiling clock with a given name.

    Outputs a unique ID that must be passed to EndClock.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Clock A"}),
                "cache": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Cache the clock ID, this means the node will follow Comfy's default invalidation system. If False it will always invalidate / mark the node as 'dirty'",
                    },
                ),
            },
            "optional": {
                "passthrough": (CIO.ANY,),
            },
        }

    RETURN_TYPES = (
        CIO.ANY,
        "STRING",
    )
    RETURN_NAMES = (
        "passthrough",
        "clock_id",
    )
    FUNCTION = "start_timer"
    CATEGORY = "mtb/utils"

    def start_timer(
        self, *, name: str, passthrough: Any | None = None, **kwargs
    ):
        global active_timers

        if len(active_timers) >= MAX_CLOCKS:
            # get oldest clock
            removed_key = None
            for key, clock_data in active_timers.items():
                if clock_data["end"] is not None:
                    removed_key = key
                    break
            if removed_key:
                removed_clock = active_timers.pop(removed_key)
                log.info(
                    f"[Profiling] Evicted finished clock '{removed_clock['name']}' (ID: {removed_key}) due to limit ({MAX_CLOCKS})."
                )
            else:
                removed_key, removed_clock = active_timers.popitem(last=False)
                log.warning(
                    f"[Profiling] Evicted running clock '{removed_clock['name']}' (ID: {removed_key}) due to limit ({MAX_CLOCKS})."
                )

        clock_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        active_timers[clock_id] = {
            "start": start_time,
            "name": name,
            "end": None,
        }

        active_timers.move_to_end(clock_id)

        log.debug(f"[Profiling] Clock '{name}' (ID: {clock_id}) started.")

        return (
            passthrough,
            clock_id,
        )

    @classmethod
    def IS_CHANGED(
        cls, *, name: str, cache: bool = False, passthrough: Any | None = None
    ):
        if not cache:
            return float("Nan")

        return {"name": name, "cache": cache, "passthrough": passthrough}


class MTB_EndClock:
    """
    Stops a profiling clock identified by its ID and returns the elapsed time in milliseconds.

    Errors if the clock ID is not found or already stopped.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clock_id": (
                    "STRING",
                    {"forceInput": True},
                ),
            },
            "optional": {
                "passthrough": (CIO.ANY,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        CIO.ANY,
        "STRING",
        "FLOAT",
        "INT",
    )
    RETURN_NAMES = (
        "passthrough",
        "name",
        "seconds",
        "milliseconds",
    )
    FUNCTION = "end_timer"
    CATEGORY = "mtb/utils"

    def end_timer(self, clock_id: str, passthrough, unique_id=None):
        global active_timers

        if clock_id not in active_timers:
            raise ValueError(
                f"Error: Clock with ID '{clock_id}' not found. "
                "Ensure StartClock was executed for this ID and proper passthrough chaining."
            )

        clock = active_timers[clock_id]
        if clock.get("end") is not None:
            return (passthrough, clock["name"], clock["end"])

        start_time = clock["start"]
        end_time = time.perf_counter()

        duration_seconds = end_time - start_time
        duration_ms = int(duration_seconds * 1000)
        clock["end"] = duration_ms

        active_timers.move_to_end(clock_id)

        log.debug(
            f"[Profiling] Clock '{clock['name']}' (ID: {clock_id}) stopped. Elapsed: {duration_ms}ms"
        )
        if unique_id:
            PromptServer.instance.send_progress_text(
                f"Clock '{clock['name']}' took {duration_seconds:.4f} seconds",
                unique_id,
            )

        return (passthrough, clock["name"], duration_seconds, duration_ms)


__nodes__ = [MTB_StartClock, MTB_EndClock]
