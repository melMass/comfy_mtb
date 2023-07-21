import argparse
import json
from PIL import Image, PngImagePlugin
from rich.console import Console
from rich import print
from rich_argparse import RichHelpFormatter
import os
from pathlib import Path

console = Console()

# BNK_CutoffSetRegions
# BNK_CutoffRegionsToConditioning
# BNK_CutoffBasePrompt


# Extracts metadata from a PNG image and returns it as a dictionary
def extract_metadata(image_path):
    image = Image.open(image_path)
    prompt = image.info.get("prompt", "")
    workflow = image.info.get("workflow", "")

    if workflow:
        workflow = json.loads(workflow)

    if prompt:
        prompt = json.loads(prompt)

    console.print(f"Metadata extracted from [cyan]{image_path}[/cyan].")

    return {
        "prompt": prompt,
        "workflow": workflow,
    }


# Embeds metadata into a PNG image
def embed_metadata(image_path, metadata):
    image = Image.open(image_path)
    o_metadata = image.info

    pnginfo = PngImagePlugin.PngInfo()
    if prompt := metadata.get("prompt"):
        pnginfo.add_text("prompt", json.dumps(prompt))
    elif "prompt" in o_metadata:
        pnginfo.add_text("prompt", o_metadata["prompt"])

    if workflow := metadata.get("workflow"):
        pnginfo.add_text("workflow", json.dumps(workflow))
    elif "workflow" in o_metadata:
        pnginfo.add_text("workflow", o_metadata["workflow"])

    imgp = Path(image_path)
    output = imgp.with_stem(f"{imgp.stem}_comfy_embed")
    index = 1
    while output.exists():
        output = imgp.with_stem(f"{imgp.stem}_{index}_comfy_embed").with_suffix(".png")
        index += 1

    image.save(output, pnginfo=pnginfo)
    console.print(f"Metadata embedded into [cyan]{output}[/cyan].")


# CLI subcommand: extract
def extract(args):
    input_files = []
    for input_path in args.input:
        if os.path.isdir(input_path):
            folder_path = input_path
            input_files.extend(
                [
                    os.path.join(folder_path, file_name)
                    for file_name in os.listdir(folder_path)
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
        else:
            input_files.append(input_path)

    if len(input_files) == 1:
        metadata = extract_metadata(input_files[0])
        if args.print_output:
            print(json.dumps(metadata, indent=4))
        else:
            if not args.output:
                output = Path(input_files[0]).with_suffix(".json")
                index = 1
                while output.exists():
                    output = (
                        Path(input_files[0])
                        .with_stem(f"{Path(input_files[0]).stem}_{index}")
                        .with_suffix(".json")
                    )
                    index += 1
            else:
                output = args.output
            with open(output, "w") as file:
                json.dump(metadata, file, indent=4)
            console.print(f"Metadata extracted and saved to [cyan]{output}[/cyan].")
    else:
        metadata_dict = {}
        for input_file in input_files:
            metadata = extract_metadata(input_file)
            filename = os.path.basename(input_file)
            output = (
                Path(args.output) / f"{filename}.json"
                if args.output
                else Path(input_file).with_suffix(".json")
            )
            index = 1
            while output.exists():
                output = Path(args.output).parent / f"{filename}_{index}.json"
                index += 1
            with open(output, "w") as file:
                json.dump(metadata, file, indent=4)
            metadata_dict[filename] = metadata
        if args.output:
            with open(args.output, "w") as file:
                json.dump(metadata_dict, file, indent=4)
            console.print(
                f"Metadata extracted and saved to [cyan]{args.output}[/cyan]."
            )
        else:
            console.print("Multiple metadata files created.")


# CLI subcommand: embed
def embed(args):
    input_files = []
    for input_path in args.input:
        if os.path.isdir(input_path):
            folder_path = input_path
            input_files.extend(
                [
                    os.path.join(folder_path, file_name)
                    for file_name in os.listdir(folder_path)
                    if file_name.lower().endswith(".json")
                ]
            )
        else:
            input_files.append(input_path)

    for input_file in input_files:
        with open(input_file) as file:
            metadata = json.load(file)
        image_path = input_file.replace(".json", ".png")
        if args.output:
            output_dir = args.output
            if os.path.isdir(output_dir):
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                index = 1
                while os.path.exists(output_path):
                    output_path = os.path.join(
                        output_dir,
                        f"{os.path.basename(image_path)}_{index}.png",
                    )
                    index += 1
            else:
                output_path = output_dir
        else:
            output_path = image_path.replace(".png", "_comfy_embed.png")

        embed_metadata(image_path, metadata)
        # os.rename(image_path, output_path)
        console.print(f"Metadata embedded into [cyan]{output_path}[/cyan].")


if __name__ == "__main__":
    # Create the main CLI parser
    parser = argparse.ArgumentParser(
        prog="image-metadata-cli", formatter_class=RichHelpFormatter
    )
    subparsers = parser.add_subparsers(title="subcommands")

    # Parser for the "extract" subcommand
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract metadata from PNG image(s) or folder",
        formatter_class=RichHelpFormatter,
    )
    extract_parser.add_argument(
        "input", nargs="+", help="Input PNG image file(s) or folder path"
    )
    extract_parser.add_argument(
        "--print",
        dest="print_output",
        action="store_true",
        help="Print the output to stdout",
    )
    extract_parser.add_argument("--output", help="Output JSON file(s) or directory")
    extract_parser.set_defaults(func=extract)

    # Parser for the "embed" subcommand
    embed_parser = subparsers.add_parser(
        "embed",
        help="Embed metadata into PNG image(s) or folder",
        formatter_class=RichHelpFormatter,
    )
    embed_parser.add_argument(
        "input", nargs="+", help="Input JSON file(s) or folder path"
    )
    embed_parser.add_argument("--output", help="Output PNG image file(s) or directory")
    embed_parser.set_defaults(func=embed)

    # Parse the command-line arguments and execute the appropriate subcommand
    args = parser.parse_args()
    if hasattr(args, "func"):
        try:
            args.func(args)
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
    else:
        parser.print_help()
