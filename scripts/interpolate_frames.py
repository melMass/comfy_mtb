import glob
from pathlib import Path
import uuid
import sys
from typing import List

sys.path.append((Path(__file__).parent / "extern").as_posix())


import argparse
from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.progress import Progress

import numpy as np
import subprocess


def write_prores_444_video(output_file, frames: List[np.ndarray], fps):
    # Convert float images to the range of 0-65535 (12-bit color depth)
    frames = [(frame * 65535).clip(0, 65535).astype(np.uint16) for frame in frames]

    height, width, _ = frames[0].shape

    # Prepare the FFmpeg command
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it already exists
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb48le",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4",
        "-pix_fmt",
        "yuva444p10le",
        "-r",
        str(fps),
        "-y",  # Overwrite output file if it already exists
        output_file,
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    for frame in frames:
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()


if __name__ == "__main__":
    default_output = f"./output_{uuid.uuid4()}.mov"
    parser = argparse.ArgumentParser(
        description="FILM frame interpolation", formatter_class=RichHelpFormatter
    )
    parser.add_argument("inputs", nargs="*", help="Input image files")
    parser.add_argument("--output", help="Output JSON file", default=default_output)
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--glob", help="Enable glob pattern matching", metavar="PATTERN"
    )
    parser.add_argument(
        "--interpolate", type=int, default=4, help="Time for interpolated frames"
    )
    parser.add_argument("--fps", type=int, default=30, help="Out FPS")
    align = 64
    block_width = 2
    block_height = 2

    args = parser.parse_args()

    # - checks
    if not args.glob and not args.inputs:
        parser.error("Either --glob flag or inputs must be provided.")
    if args.glob:
        glob_pattern = args.glob
        try:
            pattern_path = str(Path(glob_pattern).expanduser().resolve())

            if not any(glob.glob(pattern_path)):
                raise ValueError(f"No files found for glob pattern: {glob_pattern}")
        except Exception as e:
            console = Console()
            console.print(
                f"[bold red]Error: Invalid glob pattern '{glob_pattern}': {e}[/bold red]"
            )

            exit(1)
    else:
        glob_pattern = None

    input_files: List[Path] = []

    if glob_pattern:
        input_files = [
            Path(p)
            for p in list(glob.glob(str(Path(glob_pattern).expanduser().resolve())))
        ]
    else:
        input_files = [Path(p) for p in args.inputs]

    console = Console()
    console.print("Input Files:", style="bold", end=" ")
    console.print(f"{len(input_files):03d} files", style="cyan")
    # for input_file in args.inputs:
    #     console.print(f"- {input_file}", style="cyan")
    console.print("\nOutput File:", style="bold", end=" ")
    console.print(f"{Path(args.output).resolve().absolute()}", style="cyan")

    with Progress(console=console, auto_refresh=True) as progress:
        from frame_interpolation.eval import util
        from frame_interpolation.eval import util, interpolator

        # files = Path(pth).rglob("*.png")

        model = interpolator.Interpolator(
            "G:/MODELS/FILM/pretrained_models/film_net/Style", None
        )  # [2,2]

        task = progress.add_task("[cyan]Interpolating frames...", total=1)

        frames = list(
            util.interpolate_recursively_from_files(
                [x.as_posix() for x in input_files], args.interpolate, model
            )
        )

        # mediapy.write_video(args.output, frames, fps=args.fps)
        write_prores_444_video(args.output, frames, fps=args.fps)
        progress.update(task, advance=1)
        progress.refresh()
