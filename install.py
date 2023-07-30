import requests
import os
import ast
import re
import argparse
import sys
import subprocess
from importlib import import_module
import platform
from pathlib import Path
import sys
import zipfile
import shutil
import stat


here = Path(__file__).parent
executable = sys.executable

# - detect mode
mode = None
if os.environ.get("COLAB_GPU"):
    mode = "colab"
elif "python_embeded" in executable:
    mode = "embeded"
elif ".venv" in executable:
    mode = "venv"


if mode == None:
    mode = "unknown"

# region ansi
# ANSI escape sequences for text styling
ANSI_FORMATS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "blink": "\033[5m",
    "reverse": "\033[7m",
    "strike": "\033[9m",
}

ANSI_COLORS = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[30;1m",
    "bright_red": "\033[31;1m",
    "bright_green": "\033[32;1m",
    "bright_yellow": "\033[33;1m",
    "bright_blue": "\033[34;1m",
    "bright_magenta": "\033[35;1m",
    "bright_cyan": "\033[36;1m",
    "bright_white": "\033[37;1m",
    "bg_black": "\033[40m",
    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
    "bg_white": "\033[47m",
    "bg_bright_black": "\033[40;1m",
    "bg_bright_red": "\033[41;1m",
    "bg_bright_green": "\033[42;1m",
    "bg_bright_yellow": "\033[43;1m",
    "bg_bright_blue": "\033[44;1m",
    "bg_bright_magenta": "\033[45;1m",
    "bg_bright_cyan": "\033[46;1m",
    "bg_bright_white": "\033[47;1m",
}


def apply_format(text, *formats):
    """Apply ANSI escape sequences for the specified formats to the given text."""
    formatted_text = text
    for format in formats:
        formatted_text = f"{ANSI_FORMATS.get(format, '')}{formatted_text}{ANSI_FORMATS.get('reset', '')}"
    return formatted_text


def apply_color(text, color=None, background=None):
    """Apply ANSI escape sequences for the specified color and background to the given text."""
    formatted_text = text
    if color:
        formatted_text = f"{ANSI_COLORS.get(color, '')}{formatted_text}{ANSI_FORMATS.get('reset', '')}"
    if background:
        formatted_text = f"{ANSI_COLORS.get(background, '')}{formatted_text}{ANSI_FORMATS.get('reset', '')}"
    return formatted_text


def print_formatted(text, *formats, color=None, background=None, **kwargs):
    """Print the given text with the specified formats, color, and background."""
    formatted_text = apply_format(text, *formats)
    formatted_text = apply_color(formatted_text, color, background)
    file = kwargs.get("file", sys.stdout)
    print(
        apply_color(apply_format("[mtb install] ", "bold"), color="yellow"),
        formatted_text,
        file=file,
    )


# endregion

try:
    import requirements
except ImportError:
    print_formatted("Installing requirements-parser...", "italic", color="yellow")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "requirements-parser"]
    )
    import requirements

    print_formatted("Done.", "italic", color="green")

try:
    from tqdm import tqdm
except ImportError:
    print_formatted("Installing tqdm...", "italic", color="yellow")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "tqdm"])
    from tqdm import tqdm
import importlib


pip_map = {
    "onnxruntime-gpu": "onnxruntime",
    "opencv-contrib": "cv2",
    "tb-nightly": "tensorboard",
    "protobuf": "google.protobuf",
    # Add more mappings as needed
}


def is_pipe():
    try:
        mode = os.fstat(0).st_mode
        return (
            stat.S_ISFIFO(mode)
            or stat.S_ISREG(mode)
            or stat.S_ISBLK(mode)
            or stat.S_ISSOCK(mode)
        )
    except OSError:
        return False


# Get the version from __init__.py
def get_local_version():
    init_file = os.path.join(os.path.dirname(__file__), "__init__.py")
    if os.path.isfile(init_file):
        with open(init_file, "r") as f:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "__version__"
                            and isinstance(node.value, ast.Str)
                        ):
                            return node.value.s
    return None


def download_file(url, file_name):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with open(file_name, "wb") as file, tqdm(
            desc=file_name.stem,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))


def get_requirements(path: Path):
    with open(path.resolve(), "r") as requirements_file:
        requirements_txt = requirements_file.read()

    try:
        parsed_requirements = requirements.parse(requirements_txt)
    except AttributeError:
        print_formatted(
            f"Failed to parse {path}. Please make sure the file is correctly formatted.",
            "bold",
            color="red",
        )

        return

    return parsed_requirements


def try_import(requirement):
    dependency = requirement.name.strip()
    import_name = pip_map.get(dependency, dependency)
    installed = False

    pip_name = dependency
    if specs := requirement.specs:
        pip_name += "".join(specs[0])

    try:
        import_module(import_name)
        print_formatted(
            f"Package {pip_name} already installed (import name: '{import_name}').",
            "bold",
            color="green",
        )
        installed = True
    except ImportError:
        pass

    return (installed, pip_name, import_name)


def import_or_install(requirement, dry=False):
    installed, pip_name, import_name = try_import(requirement)

    if not installed:
        print_formatted(f"Installing package {pip_name}...", "italic", color="yellow")
        if dry:
            print_formatted(
                f"Dry-run: Package {pip_name} would be installed (import name: '{import_name}').",
                color="yellow",
            )
        else:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pip_name]
                )
                print_formatted(
                    f"Package {pip_name} installed successfully using pip package name  (import name: '{import_name}')",
                    "bold",
                    color="green",
                )
            except subprocess.CalledProcessError as e:
                print_formatted(
                    f"Failed to install package {pip_name} using pip package name  (import name: '{import_name}'). Error: {str(e)}",
                    "bold",
                    color="red",
                )


# Install dependencies from requirements.txt
def install_dependencies(dry=False):
    parsed_requirements = get_requirements(here / "reqs.txt")
    if not parsed_requirements:
        return
    print_formatted(
        "Installing dependencies from reqs.txt...", "italic", color="yellow"
    )

    for requirement in parsed_requirements:
        import_or_install(requirement, dry=dry)


if __name__ == "__main__":
    full = False
    if is_pipe():
        print_formatted("Pipe detected, full install...", color="green")
        # we clone our repo
        url = "https://github.com/melmass/comfy_mtb.git"
        clone_dir = here / "custom_nodes" / "comfy_mtb"
        if not clone_dir.exists():
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
            print_formatted(f"Cloning {url} to {clone_dir}", "italic", color="yellow")
            subprocess.check_call(["git", "clone", "--recursive", url, clone_dir])

        # os.chdir(clone_dir)
        here = clone_dir
        full = True

    if len(sys.argv) == 1:
        print_formatted(
            "No arguments provided, doing a full install/update...",
            "italic",
            color="yellow",
        )

        full = True

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wheels", "-w", action="store_true", help="Install wheel dependencies"
    )
    parser.add_argument(
        "--requirements", "-r", action="store_true", help="Install requirements.txt"
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Print what will happen without doing it (still making requests to the GH Api)",
    )

    # parser.add_argument(
    #     "--version",
    #     default=get_local_version(),
    #     help="Version to check against the GitHub API",
    # )

    args = parser.parse_args()

    wheels_directory = here / "wheels"
    print_formatted(f"Detected environment: {apply_color(mode,'cyan')}")

    # Install dependencies from requirements.txt
    # if args.requirements or mode == "venv":

    if (not args.wheels and mode not in ["colab", "embeded"]) and not full:
        print_formatted(
            "Skipping wheel installation. Use --wheels to install wheel dependencies. (only needed for Comfy embed)",
            "italic",
            color="yellow",
        )

        install_dependencies(dry=args.dry)
        sys.exit()

    if mode in ["colab", "embeded"]:
        print_formatted(
            f"Downloading and installing release wheels since we are in a Comfy {apply_color(mode,'cyan')} environment",
        )
    if full:
        print_formatted(
            f"Downloading and installing release wheels since no arguments where provided"
        )

    # - Check the env before proceeding.
    missing_wheels = False
    parsed_requirements = get_requirements(here / "reqs.txt")
    if parsed_requirements:
        for requirement in parsed_requirements:
            installed, pip_name, import_name = try_import(requirement)
            if not installed:
                missing_wheels = True
                break

    if not missing_wheels:
        print_formatted(
            f"All requirements are already installed.", "italic", color="green"
        )
        sys.exit()

    # Fetch the JSON data from the GitHub API URL
    owner = "melmass"
    repo = "comfy_mtb"
    # version = args.version
    current_platform = platform.system().lower()

    # Get the tag version from the GitHub API
    tag_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    response = requests.get(tag_url)
    if response.status_code == 404:
        # print_formatted(
        #     f"Tag version '{apply_color(version,'cyan')}' not found for {owner}/{repo} repository."
        # )
        print_formatted("Error retrieving the release assets.", color="red")
        sys.exit()

    tag_data = response.json()
    tag_name = tag_data["name"]

    # # Compare the local and tag versions
    # if version and tag_name:
    #     if re.match(r"v?(\d+(\.\d+)+)", version) and re.match(
    #         r"v?(\d+(\.\d+)+)", tag_name
    #     ):
    #         version_parts = [int(part) for part in version.lstrip("v").split(".")]
    #         tag_version_parts = [int(part) for part in tag_name.lstrip("v").split(".")]

    #         if version_parts > tag_version_parts:
    #             print_formatted(
    #                 f"Local version ({version}) is greater than the release version ({tag_name}).",
    #                 "bold",
    #                 "yellow",
    #             )
    #             sys.exit()

    # Download the assets for the given version
    matching_assets = [
        asset
        for asset in tag_data["assets"]
        if current_platform in asset["name"] and asset["name"].endswith("zip")
    ]
    if not matching_assets:
        print_formatted(
            f"Unsupported operating system: {current_platform}", color="yellow"
        )

    wheels_directory.mkdir(exist_ok=True)
    # - Install the wheels
    for asset in matching_assets:
        asset_name = asset["name"]
        asset_download_url = asset["browser_download_url"]
        print_formatted(f"Downloading asset: {asset_name}", color="yellow")
        asset_dest = wheels_directory / asset_name
        download_file(asset_download_url, asset_dest)

        # - Unzip to wheels dir
        whl_files = []
        whl_order = None
        with zipfile.ZipFile(asset_dest, "r") as zip_ref:
            for item in tqdm(zip_ref.namelist(), desc="Extracting", unit="file"):
                if item.endswith(".whl"):
                    item_basename = os.path.basename(item)
                    target_path = wheels_directory / item_basename
                    with zip_ref.open(item) as source, open(
                        target_path, "wb"
                    ) as target:
                        whl_files.append(target_path)
                        shutil.copyfileobj(source, target)
                elif item.endswith("order.txt"):
                    item_basename = os.path.basename(item)
                    target_path = wheels_directory / item_basename
                    with zip_ref.open(item) as source, open(
                        target_path, "wb"
                    ) as target:
                        whl_order = target_path
                        shutil.copyfileobj(source, target)

        print_formatted(
            f"Wheels extracted for {current_platform} to the '{wheels_directory}' directory.",
            "bold",
            color="green",
        )

        if whl_files:
            if whl_order:
                with open(whl_order, "r") as order:
                    wheel_order_lines = [line.strip() for line in order]
                whl_files = sorted(
                    whl_files,
                    key=lambda x: wheel_order_lines.index(x.name.split("-")[0]),
                )

            for whl_file in tqdm(whl_files, desc="Installing", unit="package"):
                whl_path = wheels_directory / whl_file

                # check if installed
                try:
                    whl_dep = whl_path.name.split("-")[0]
                    import_name = pip_map.get(whl_dep, whl_dep)
                    import_module(import_name)
                    tqdm.write(
                        f"Package {import_name} already installed, skipping wheel installation.",
                    )
                    continue
                except ImportError:
                    if args.dry:
                        tqdm.write(
                            f"Dry-run: Package {whl_path.name} would be installed.",
                        )
                        continue

                    tqdm.write("Installing wheel: " + whl_path.name)

                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            whl_path.as_posix(),
                        ]
                    )

            print_formatted("Wheels installation completed.", color="green")
        else:
            print_formatted("No .whl files found. Nothing to install.", color="yellow")

    # - Install all remainings
    install_dependencies(dry=args.dry)
