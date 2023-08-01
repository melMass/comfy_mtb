import requests
import os
import ast
import argparse
import sys
import subprocess
from importlib import import_module
import platform
from pathlib import Path
import sys
import stat
import threading
import signal
from contextlib import suppress
from queue import Queue, Empty
from contextlib import contextmanager

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


if mode is None:
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
    header = "[mtb install] "

    # Handle console encoding for Unicode characters (utf-8)
    encoded_header = header.encode("utf-8", errors="replace").decode("utf-8")
    encoded_text = formatted_text.encode("utf-8", errors="replace").decode("utf-8")

    if sys.platform == "win32":
        output_text = (
            " " * len(encoded_header)
            if kwargs.get("no_header")
            else apply_color(apply_format(encoded_header, "bold"), color="yellow")
        )
        output_text += encoded_text + "\n"
        sys.stdout.buffer.write(output_text.encode("utf-8"))
    else:
        print(
            " " * len(encoded_header)
            if kwargs.get("no_header")
            else apply_color(apply_format(encoded_header, "bold"), color="yellow"),
            encoded_text,
            file=file,
        )


# endregion


# region utils
def enqueue_output(out, queue):
    for line in iter(out.readline, b""):
        queue.put(line)
    out.close()


def run_command(cmd):
    if isinstance(cmd, str):
        shell_cmd = cmd
        shell = True
    elif isinstance(cmd, list):
        shell_cmd = " ".join(cmd)
        shell = False
    else:
        raise ValueError(
            "Invalid 'cmd' argument. It must be a string or a list of arguments."
        )

    process = subprocess.Popen(
        shell_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
    )

    # Create separate threads to read standard output and standard error streams
    stdout_queue = Queue()
    stderr_queue = Queue()
    stdout_thread = threading.Thread(
        target=enqueue_output, args=(process.stdout, stdout_queue)
    )
    stderr_thread = threading.Thread(
        target=enqueue_output, args=(process.stderr, stderr_queue)
    )
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    interrupted = False

    def signal_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("Command execution interrupted.")

    # Register the signal handler for keyboard interrupts (SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    # Process output from both streams until the process completes or interrupted
    while not interrupted and (
        process.poll() is None or not stdout_queue.empty() or not stderr_queue.empty()
    ):
        with suppress(Empty):
            stdout_line = stdout_queue.get_nowait()
            if stdout_line.strip() != "":
                print(stdout_line.strip())
        with suppress(Empty):
            stderr_line = stderr_queue.get_nowait()
            if stderr_line.strip() != "":
                print(stderr_line.strip())
    return_code = process.returncode

    if return_code == 0 and not interrupted:
        print("Command executed successfully!")
    else:
        if not interrupted:
            print(f"Command failed with return code: {return_code}")


# endregion

try:
    import requirements
except ImportError:
    print_formatted("Installing requirements-parser...", "italic", color="yellow")
    run_command([sys.executable, "-m", "pip", "install", "requirements-parser"])
    import requirements

    print_formatted("Done.", "italic", color="green")

try:
    from tqdm import tqdm
except ImportError:
    print_formatted("Installing tqdm...", "italic", color="yellow")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "tqdm"])
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
    if not sys.stdin.isatty():
        return False
    if sys.platform == "win32":
        try:
            import msvcrt

            return msvcrt.get_osfhandle(0) != -1
        except ImportError:
            return False
    else:
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


@contextmanager
def suppress_std():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull

        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


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
    pip_spec = "".join(specs[0]) if (specs := requirement.specs) else ""
    try:
        with suppress_std():
            import_module(import_name)
        print_formatted(
            f"\t✅ Package {pip_name} already installed (import name: '{import_name}').",
            "bold",
            color="green",
            no_header=True,
        )
        installed = True
    except ImportError:
        print_formatted(
            f"\t⛔ Package {pip_name} is missing (import name: '{import_name}').",
            "bold",
            color="red",
            no_header=True,
        )

    return (installed, pip_name, pip_spec, import_name)


def import_or_install(requirement, dry=False):
    installed, pip_name, pip_spec, import_name = try_import(requirement)

    pip_install_name = pip_name + pip_spec

    if not installed:
        print_formatted(f"Installing package {pip_name}...", "italic", color="yellow")
        if dry:
            print_formatted(
                f"Dry-run: Package {pip_install_name} would be installed (import name: '{import_name}').",
                color="yellow",
            )
        else:
            try:
                run_command([sys.executable, "-m", "pip", "install", pip_install_name])
                print_formatted(
                    f"Package {pip_install_name} installed successfully using pip package name  (import name: '{import_name}')",
                    "bold",
                    color="green",
                )
            except subprocess.CalledProcessError as e:
                print_formatted(
                    f"Failed to install package {pip_install_name} using pip package name  (import name: '{import_name}'). Error: {str(e)}",
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
            run_command(["git", "clone", "--recursive", url, clone_dir.as_posix()])

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

    # wheels_directory = here / "wheels"
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
            "italic",
            color="yellow",
        )
    if full:
        print_formatted(
            f"Downloading and installing release wheels since no arguments where provided",
            "italic",
            color="yellow",
        )

    # - Check the env before proceeding.
    missing_deps = []
    parsed_requirements = get_requirements(here / "reqs.txt")
    if parsed_requirements:
        for requirement in parsed_requirements:
            installed, pip_name, pip_spec, import_name = try_import(requirement)
            if not installed:
                missing_deps.append(pip_name.split("-")[0])

    if len(missing_deps) == 0:
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

    short_platform = {
        "windows": "win_amd64",
        "linux": "linux_x86_64",
    }
    matching_assets = [
        asset
        for asset in tag_data["assets"]
        if asset["name"].endswith(".whl")
        and (
            "any" in asset["name"] or short_platform[current_platform] in asset["name"]
        )
    ]
    if not matching_assets:
        print_formatted(
            f"Unsupported operating system: {current_platform}", color="yellow"
        )
    wheel_order_asset = next(
        (asset for asset in tag_data["assets"] if asset["name"] == "wheel_order.txt"),
        None,
    )
    if wheel_order_asset is not None:
        print_formatted(
            "⚙️ Sorting the release wheels using wheels order", "italic", color="yellow"
        )
        response = requests.get(wheel_order_asset["browser_download_url"])
        if response.status_code == 200:
            wheel_order = [line.strip() for line in response.text.splitlines()]

            def get_order_index(val):
                try:
                    return wheel_order.index(val)
                except ValueError:
                    return len(wheel_order)

            matching_assets = sorted(
                matching_assets,
                key=lambda x: get_order_index(x["name"].split("-")[0]),
            )
        else:
            print("Failed to fetch wheel_order.txt. Status code:", response.status_code)

    missing_deps_urls = []
    for whl_file in matching_assets:
        # check if installed
        whl_dep = whl_file["name"].split("-")[0]
        missing_deps_urls.append(whl_file["browser_download_url"])

        # run_command(
        #     [
        #         sys.executable,
        #         "-m",
        #         "pip",
        #         "install",
        #         whl_path.as_posix(),
        #     ]
        # )
        # # - Install the wheels
        # for asset in matching_assets:
        #     asset_name = asset["name"]
        #     asset_download_url = asset["browser_download_url"]
        #     print_formatted(f"Downloading asset: {asset_name}", color="yellow")
        #     asset_dest = wheels_directory / asset_name
        #     download_file(asset_download_url, asset_dest)

        #     # - Unzip to wheels dir
        #     whl_files = []
        #     whl_order = None
        #     with zipfile.ZipFile(asset_dest, "r") as zip_ref:
        #         for item in tqdm(zip_ref.namelist(), desc="Extracting", unit="file"):
        #             if item.endswith(".whl"):
        #                 item_basename = os.path.basename(item)
        #                 target_path = wheels_directory / item_basename
        #                 with zip_ref.open(item) as source, open(
        #                     target_path, "wb"
        #                 ) as target:
        #                     whl_files.append(target_path)
        #                     shutil.copyfileobj(source, target)
        #             elif item.endswith("order.txt"):
        #                 item_basename = os.path.basename(item)
        #                 target_path = wheels_directory / item_basename
        #                 with zip_ref.open(item) as source, open(
        #                     target_path, "wb"
        #                 ) as target:
        #                     whl_order = target_path
        #                     shutil.copyfileobj(source, target)

        #     print_formatted(
        #         f"Wheels extracted for {current_platform} to the '{wheels_directory}' directory.",
        #         "bold",
        #         color="green",
        #     )

    # print_formatted(
    #     "\tFound those missing wheels from the release:\n\t\t -"
    #     + "\n\t\t - ".join(missing_deps_urls),
    #     "italic",
    #     color="yellow",
    #     no_header=True,
    # )

    install_cmd = [sys.executable, "-m", "pip", "install"]

    wheel_cmd = install_cmd + missing_deps_urls

    # - Install all deps
    if not args.dry:
        run_command(wheel_cmd)
        run_command(install_cmd + ["-r", (here / "reqs.txt").as_posix()])
        print_formatted(
            "Successfully installed all dependencies.", "italic", color="green"
        )
    else:
        print_formatted(
            f"Would have run the following command:\n\t{apply_color(' '.join(install_cmd),'cyan')}",
            "italic",
            color="yellow",
        )
