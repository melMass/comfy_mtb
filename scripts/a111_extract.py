from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngImageFile, PngInfo
import json
from pprint import pprint
import argparse
from rich.console import Console
from rich.progress import Progress
from rich_argparse import RichHelpFormatter


def parse_a111(params, verbose=False):
    # params = [p.split(": ") for p in params.split("\n")]
    params = params.split("\n")

    prompt = params[0].strip()
    neg = params[1].split(":")[1].strip()

    settings = {}
    try:
        settings = {
            s.split(":")[0].strip(): s.split(":")[1].strip()
            for s in params[2].split(",")
        }

    except IndexError:
        settings = {"raw": params[2].strip()}

    if verbose:
        print(f"PROMPT: {prompt}")
        print(f"NEG: {neg}")
        print("SETTINGS:")
        pprint(settings, indent=4)

    return {"prompt": prompt, "negative": neg, "settings": settings}


import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crude metadata extractor from A111 pngs",
        formatter_class=RichHelpFormatter
    )
    parser.add_argument("inputs", nargs="*", help="Input image files")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument(
        "--glob", help="Enable glob pattern matching", metavar="PATTERN"
    )

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

    input_files = []

    if glob_pattern:
        input_files = list(glob.glob(str(Path(glob_pattern).expanduser().resolve())))
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
        # files = Path(pth).rglob("*.png")
        unique_info = {}
        last = None

        task = progress.add_task("[cyan]Extracting meta...", total=len(input_files) + 1)
        for p in input_files:
            im = Image.open(p)
            parsed = parse_a111(im.info["parameters"], args.verbose)

            if parsed != last:
                unique_info[Path(p).stem] = parsed

            last = parsed
            progress.update(task, advance=1)
            progress.refresh()

        unique_info = json.dumps(unique_info, indent=4)
        with open(args.output, "w") as f:
            f.write(unique_info)
            progress.update(task, advance=1)
            progress.refresh()

    console.print("\nProcessing completed!", style="bold green")
