import os
import requests
from rich.console import Console
from tqdm import tqdm
import subprocess
import sys

try:
    import folder_paths
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
    import folder_paths

models_to_download = {
    "DeepBump": {
        "size": 25.5,
        "download_url": "https://github.com/HugoTini/DeepBump/raw/master/deepbump256.onnx",
        "destination": "deepbump",
    },
    "Face Swap": {
        "size": 660,
        "download_url": [
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth",
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx",
        ],
        "destination": "insightface",
    },
    "GFPGAN (face enhancement)": {
        "size": 332,
        "download_url": [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            # TODO: provide a way to selectively download models from "packs"
            # https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth
            # https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth
            # https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth
        ],
        "destination": "face_restore",
    },
    "FILM: Frame Interpolation for Large Motion": {
        "size": 402,
        "download_url": [
            "https://drive.google.com/drive/folders/131_--QrieM4aQbbLWrUtbO2cGbX8-war"
        ],
        "destination": "FILM",
    },
}

console = Console()

from urllib.parse import urlparse
from pathlib import Path


def download_model(download_url, destination):
    if isinstance(download_url, list):
        for url in download_url:
            download_model(url, destination)
        return

    filename = os.path.basename(urlparse(download_url).path)
    response = None
    if "drive.google.com" in download_url:
        try:
            import gdown
        except ImportError:
            print("Installing gdown")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "git+https://github.com/melMass/gdown@main",
                ]
            )
            import gdown

        if "/folders/" in download_url:
            # download folder
            try:
                gdown.download_folder(download_url, output=destination, resume=True)
            except TypeError:
                gdown.download_folder(download_url, output=destination)

            return
        # download from google drive
        gdown.download(download_url, destination, quiet=False, resume=True)
        return

    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    destination_path = os.path.join(destination, filename)
    with open(destination_path, "wb") as file:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc=destination_path, ncols=80
        ) as progress_bar:
            for data in response.iter_content(chunk_size=4096):
                file.write(data)
                progress_bar.update(len(data))

    console.print(
        f"Downloaded model from {download_url} to {destination_path}",
        style="bold green",
    )


def ask_user_for_downloads(models_to_download):
    console.print("Choose models to download:")
    choices = {}
    for i, model_name in enumerate(models_to_download.keys(), start=1):
        choices[str(i)] = model_name
        console.print(f"{i}. {model_name}")

    console.print(
        "Enter the numbers of the models you want to download (comma-separated):"
    )
    user_input = console.input(">> ")
    selected_models = user_input.split(",")
    models_to_download_selected = {}

    for choice in selected_models:
        choice = choice.strip()

        if choice in choices:
            model_name = choices[choice]
            models_to_download_selected[model_name] = models_to_download[model_name]

        elif choice == "":
            # download all
            models_to_download_selected = models_to_download
        else:
            console.print(f"Invalid choice: {choice}. Skipping.")

    return models_to_download_selected


def handle_interrupt():
    console.print("Interrupted by user.", style="bold red")


def main(models_to_download, skip_input=False):
    try:
        models_to_download_selected = {}

        def check_destination(urls, destination):
            if isinstance(urls, list):
                for url in urls:
                    check_destination(url, destination)
                return

            filename = os.path.basename(urlparse(urls).path)
            destination = os.path.join(folder_paths.models_dir, destination)

            if not os.path.exists(destination):
                os.makedirs(destination)

            destination_path = os.path.join(destination, filename)
            if os.path.exists(destination_path):
                url_name = os.path.basename(urlparse(urls).path)
                console.print(
                    f"Checkpoint '{url_name}' for {model_name} already exists in '{destination}'"
                )
            else:
                model_details["destination"] = destination
                models_to_download_selected[model_name] = model_details

        for model_name, model_details in models_to_download.items():
            destination = model_details["destination"]
            download_url = model_details["download_url"]

            check_destination(download_url, destination)

        if not models_to_download_selected:
            console.print("No new models to download.")
            return

        models_to_download_selected = (
            ask_user_for_downloads(models_to_download_selected)
            if not skip_input
            else models_to_download_selected
        )

        for model_name, model_details in models_to_download_selected.items():
            download_url = model_details["download_url"]
            destination = model_details["destination"]
            console.print(f"Downloading {model_name}...")
            download_model(download_url, destination)

    except KeyboardInterrupt:
        handle_interrupt()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yes", action="store_true", help="skip user input")

    args = parser.parse_args()
    main(models_to_download, args.yes)
