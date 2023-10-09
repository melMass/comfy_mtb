import argparse
import ast
import os
import platform
import shlex
import stat
import subprocess
import sys
from contextlib import contextmanager, suppress
from importlib import import_module
from pathlib import Path
from queue import Empty, Queue

import requests

here = Path(__file__).parent
executable = Path(sys.executable)

# - detect mode
mode = None
if os.environ.get("COLAB_GPU"):
    mode = "colab"
elif "python_embeded" in str(executable):
    mode = "embeded"
elif ".venv" in str(executable):
    mode = "venv"


if mode is None:
    mode = "unknown"

# - Constants
repo_url = "https://github.com/melmass/comfy_mtb.git"
repo_owner = "melmass"
repo_name = "comfy_mtb"
short_platform = {
    "windows": "win_amd64",
    "linux": "linux_x86_64",
}
current_platform = platform.system().lower()

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
    encoded_header = header.encode(sys.stdout.encoding, errors="replace").decode(
        sys.stdout.encoding
    )
    encoded_text = formatted_text.encode(sys.stdout.encoding, errors="replace").decode(
        sys.stdout.encoding
    )

    print(
        " " * len(encoded_header)
        if kwargs.get("no_header")
        else apply_color(apply_format(encoded_header, "bold"), color="yellow"),
        encoded_text,
        file=file,
    )


# endregion


# region utils
def run_command(cmd, ignored_lines_start=None):
    if ignored_lines_start is None:
        ignored_lines_start = []

    if isinstance(cmd, str):
        shell_cmd = cmd
    elif isinstance(cmd, list):
        shell_cmd = " ".join(
            arg.as_posix() if isinstance(arg, Path) else shlex.quote(str(arg))
            for arg in cmd
        )
    else:
        raise ValueError(
            "Invalid 'cmd' argument. It must be a string or a list of arguments."
        )

    try:
        _run_command(shell_cmd, ignored_lines_start)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}", file=sys.stderr)
        print(e.stderr.strip(), file=sys.stderr)

    except KeyboardInterrupt:
        print("Command execution interrupted.")


def _run_command(shell_cmd, ignored_lines_start):
    print_formatted(f"Running {shell_cmd}", "bold")
    result = subprocess.run(
        shell_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
        check=True,
    )

    stdout_lines = result.stdout.strip().split("\n")
    stderr_lines = result.stderr.strip().split("\n")

    # Print stdout, skipping ignored lines
    for line in stdout_lines:
        if not any(line.startswith(ign) for ign in ignored_lines_start):
            print(line)

    # Print stderr
    for line in stderr_lines:
        print(line, file=sys.stderr)

    print("Command executed successfully!")


# endregion

try:
    import requirements
except ImportError:
    print_formatted("Installing requirements-parser...", "italic", color="yellow")
    run_command([executable, "-m", "pip", "install", "requirements-parser"])
    import requirements

    print_formatted("Done.", "italic", color="green")

try:
    from tqdm import tqdm
except ImportError:
    print_formatted("Installing tqdm...", "italic", color="yellow")
    run_command([executable, "-m", "pip", "install", "--upgrade", "tqdm"])
    from tqdm import tqdm

pip_map = {
    "onnxruntime-gpu": "onnxruntime",
    "opencv-contrib": "cv2",
    "tb-nightly": "tensorboard",
    "protobuf": "google.protobuf",
    "qrcode[pil]": "qrcode",
    "requirements-parser": "requirements"
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
                run_command([executable, "-m", "pip", "install", pip_install_name])
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


def get_github_assets(tag=None):
    if tag:
        tag_url = (
            f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/tags/{tag}"
        )
    else:
        tag_url = (
            f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
        )
    response = requests.get(tag_url)
    if response.status_code == 404:
        # print_formatted(
        #     f"Tag version '{apply_color(version,'cyan')}' not found for {owner}/{repo} repository."
        # )
        print_formatted("Error retrieving the release assets.", color="red")
        sys.exit()

    tag_data = response.json()
    tag_name = tag_data["name"]

    return tag_data, tag_name


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


def main():
    full = False
    if len(sys.argv) == 1:
        print_formatted(
            "mtb doesn't need an install script anymore.", "italic", color="yellow"
        )
        return

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Comfy_mtb install script")
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="Path to clone the repository to (i.e the absolute path to ComfyUI/custom_nodes)",
    )
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

    # - keep
    # parser.add_argument(
    #     "--version",
    #     default=get_local_version(),
    #     help="Version to check against the GitHub API",
    # )
    print_formatted("mtb install", "bold", color="yellow")

    args = parser.parse_args()

    # wheels_directory = here / "wheels"
    print_formatted(f"Detected environment: {apply_color(mode,'cyan')}")

    if args.path:
        clone_dir = Path(args.path)
        if not clone_dir.exists():
            print_formatted(
                "The path provided does not exist on disk... It must be pointing to ComfyUI's custom_nodes directory"
            )
            sys.exit()

        else:
            repo_dir = clone_dir / repo_name
            if not repo_dir.exists():
                print_formatted(f"Cloning to {repo_dir}...", "italic", color="yellow")
                run_command(["git", "clone", "--recursive", repo_url, repo_dir])
            else:
                print_formatted(
                    f"Directory {repo_dir} already exists, we will update it..."
                )
                run_command(["git", "pull", "-C", repo_dir])
        # os.chdir(clone_dir)
        here = clone_dir
        full = True

    # Install dependencies from requirements.txt
    # if args.requirements or mode == "venv":

    # if (not args.wheels and mode not in ["colab", "embeded"]) and not full:
    #     print_formatted(
    #         "Skipping wheel installation. Use --wheels to install wheel dependencies. (only needed for Comfy embed)",
    #         "italic",
    #         color="yellow",
    #     )

    #     install_dependencies(dry=args.dry)
    #     sys.exit()

    # if mode in ["colab", "embeded"]:
    #     print_formatted(
    #         f"Downloading and installing release wheels since we are in a Comfy {apply_color(mode,'cyan')} environment",
    #         "italic",
    #         color="yellow",
    #     )
    # if full:
    #     print_formatted(
    #         f"Downloading and installing release wheels since no arguments where provided",
    #         "italic",
    #         color="yellow",
    #     )

    print_formatted("Checking environment...", "italic", color="yellow")
    missing_deps = []
    if parsed_requirements := get_requirements(here / "reqs.txt"):
        for requirement in parsed_requirements:
            installed, pip_name, pip_spec, import_name = try_import(requirement)
            if not installed:
                missing_deps.append(pip_name.split("-")[0])

    if not missing_deps:
        print_formatted(
            "All requirements are already installed. Enjoy 🚀",
            "italic",
            color="green",
        )
        sys.exit()

    # # - Get the tag version from the GitHub API
    # tag_data, tag_name = get_github_assets(tag=None)

    # # - keep
    # version = args.version
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

    # matching_assets = [
    #     asset
    #     for asset in tag_data["assets"]
    #     if asset["name"].endswith(".whl")
    #     and (
    #         "any" in asset["name"] or short_platform[current_platform] in asset["name"]
    #     )
    # ]
    # if not matching_assets:
    #     print_formatted(
    #         f"Unsupported operating system: {current_platform}", color="yellow"
    #     )
    # wheel_order_asset = next(
    #     (asset for asset in tag_data["assets"] if asset["name"] == "wheel_order.txt"),
    #     None,
    # )
    # if wheel_order_asset is not None:
    #     print_formatted(
    #         "⚙️ Sorting the release wheels using wheels order", "italic", color="yellow"
    #     )
    #     response = requests.get(wheel_order_asset["browser_download_url"])
    #     if response.status_code == 200:
    #         wheel_order = [line.strip() for line in response.text.splitlines()]

    #         def get_order_index(val):
    #             try:
    #                 return wheel_order.index(val)
    #             except ValueError:
    #                 return len(wheel_order)

    #         matching_assets = sorted(
    #             matching_assets,
    #             key=lambda x: get_order_index(x["name"].split("-")[0]),
    #         )
    #     else:
    #         print("Failed to fetch wheel_order.txt. Status code:", response.status_code)

    # missing_deps_urls = []
    # for whl_file in matching_assets:
    #     # check if installed
    #     missing_deps_urls.append(whl_file["browser_download_url"])

    # install_cmd = [executable, "-m", "pip", "install"]

    # - Install all deps
    # if not args.dry:
    #     if platform.system() == "Windows":
    #         wheel_cmd = install_cmd + ["-r", (here / "reqs_windows.txt")]
    #     else:
    #         wheel_cmd = install_cmd + ["-r", (here / "reqs.txt")]

    #     run_command(wheel_cmd)
    #     print_formatted(
    #         "✅ Successfully installed all dependencies.", "italic", color="green"
    #     )
    # else:
    #     print_formatted(
    #         f"Would have run the following command:\n\t{apply_color(' '.join(install_cmd),'cyan')}",
    #         "italic",
    #         color="yellow",
    #     )


if __name__ == "__main__":
    main()
