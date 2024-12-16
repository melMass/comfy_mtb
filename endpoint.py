import csv
import secrets
import sys
from pathlib import Path
from typing import Any, Literal

import folder_paths
from aiohttp import web

from .log import mklog
from .utils import (
    SortMode,
    backup_file,
    build_glob_patterns,
    glob_multiple,
    here,
    import_install,
    reqs_map,
    run_command,
    styles_dir,
)

endlog = mklog("mtb endpoint")

# - ACTIONS
import asyncio
import platform
import sys
from pathlib import Path

try:
    import websockets.server
except ModuleNotFoundError:
    endlog.warning(
        "You do not have websockets installed, the video server won't work"
    )
    websockets = False

import_install("requirements")
import io

import numpy as np
from PIL import Image


def generate_random_frame():
    # Generate a random image frame
    width, height = 640, 480
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    byte_buffer = io.BytesIO()
    pil_image.save(byte_buffer, format="JPEG")
    frame_data = byte_buffer.getvalue()
    return frame_data


class VideoStreamingManager:
    def __init__(self):
        self.video_servers = {}
        self.next_port = (
            8767  # Start with a default port and increment for each server
        )

    async def start_video_streaming_server(self, video_id):
        if video_id not in self.video_servers:
            # Create and start a new video streaming server for the specified video
            video_server = await self.create_video_streaming_server(video_id)
            self.video_servers[video_id] = video_server

            return video_server

    async def video_stream(self, websocket, path):
        # Implement the logic to continuously capture and send video frames here
        while True:
            # frame_data = capture_and_encode_frame()  # Implement this function
            frame_data = generate_random_frame()
            await websocket.send(frame_data)
            await asyncio.sleep(0.033)  # Adjust the frame rate as needed

    async def create_video_streaming_server(self, video_id):
        # Create and start a new WebSocket server for the specified video
        port = self.next_port
        self.next_port += 1  # Increment port number for the next server

        server = await websockets.server.serve(
            self.video_stream, "localhost", port
        )

        return server

    async def stop_video_streaming_server(self, video_id):
        if video_id in self.video_servers:
            # Terminate and remove the video streaming server for the specified video
            video_server = self.video_servers[video_id]
            video_server.close()
            await video_server.wait_closed()
            del self.video_servers[video_id]


async def start_video_streaming_server():
    async def video_stream(websocket, path):
        # Continuously capture and send video frames here
        while True:
            frame_data = capture_and_encode_frame()  # Implement this function
            await websocket.send(frame_data)
            await asyncio.sleep(0.033)  # Adjust the frame rate as needed

    start_server = websockets.server.serve(
        video_stream, "localhost", 8766
    )  # Use a different port (e.g., 8766)

    return await start_server


def ACTIONS_installDependency(dependency_names=None):
    if dependency_names is None:
        # return web.Response(text="No dependency name provided", status=400)
        return {"error": "No dependency name provided"}

    endlog.debug(f"Received Install Dependency request for {dependency_names}")
    # reqs = []
    resolved_names = [reqs_map.get(name, name) for name in dependency_names]
    try:
        run_command(
            [Path(sys.executable), "-m", "pip", "install"] + resolved_names
        )
        return {"success": True}

    except Exception as e:
        return {"error": f"Failed to install dependencies: {e}"}

    # if platform.system() == "Windows":
    #     reqs = list(requirements.parse((here / "reqs_windows.txt").read_text()))
    # else:
    #     reqs = list(requirements.parse((here / "reqs.txt").read_text()))
    # print([x.specs for x in reqs])
    # print(
    #     "\n".join([f"{x.line} {''.join(x.specs[0] if x.specs else '')}" for x in reqs])
    # )
    # for dependency_name in dependency_names:
    #     for req in reqs:
    #         if req.name == dependency_name:
    #             endlog.debug(f"Dependency {dependency_name} installed")
    #             break


def ACTIONS_getUserImages(
    mode: Literal["input", "output"],
    count=200,
    offset=0,
    sort: str | None = None,
    include_subfolders: bool = False,
):
    # enabled = "MTB_EXPOSE" in os.environ
    # if not enabled:
    #     return {"error": "Session not authorized to getInputs"}

    imgs = {}

    input_dir = Path(folder_paths.get_input_directory())
    output_dir = Path(folder_paths.get_output_directory())

    entry_dir = input_dir if mode == "input" else output_dir
    supported = ["png", "jpg", "jpeg", "webp", "gif"]

    entries = {}
    patterns = build_glob_patterns(supported, recursive=include_subfolders)
    entries = glob_multiple(entry_dir, patterns)

    sort_mode = SortMode.from_str(sort)

    if sort_mode:
        sort_key = {
            SortMode.MODIFIED: lambda x: x.stat().st_mtime,
            SortMode.MODIFIED_REVERSE: lambda x: x.stat().st_mtime,
            SortMode.NAME: lambda x: x.name,
            SortMode.NAME_REVERSE: lambda x: x.name,
        }.get(sort_mode)
        if sort_key:
            reverse = sort_mode in (SortMode.MODIFIED, SortMode.NAME_REVERSE)
            entries = sorted(entries, key=sort_key, reverse=reverse)

    imgs = {
        img.name: (
            f"/mtb/view?filename={img.name}&width=512&type={mode}&subfolder="
            f"{img.parent.relative_to(entry_dir) if include_subfolders else ''}"
            f"&preview=&rand={secrets.randbelow(424242)}"
        )
        for i, img in enumerate(entries)
        if offset <= i < offset + count
    }
    return imgs


def ACTIONS_getStyles(style_name=None):
    from .nodes.conditions import MTB_StylesLoader

    styles = MTB_StylesLoader.options
    match_list = ["name"]
    if styles:
        filtered_styles = {
            key: value
            for key, value in styles.items()
            if not key.startswith("__") and key not in match_list
        }
        if style_name:
            return filtered_styles.get(
                style_name, {"error": "Style not found"}
            )
        return filtered_styles
    return {"error": "No styles found"}


def ACTIONS_saveStyle(data):
    # endlog.debug(f"Received Save Styles for {data.keys()}")
    # endlog.debug(data)

    styles = [f.name for f in styles_dir.iterdir() if f.suffix == ".csv"]
    target = None
    rows = []
    for fp, content in data.items():
        if fp in styles:
            endlog.debug(f"Overwriting {fp}")
            target = styles_dir / fp
            rows = content
            break

    if not target:
        endlog.warning(
            f"Could not determine the target file for {data.keys()}"
        )
        return {"error": "Could not determine the target file for the style"}

    backup_file(target)

    with target.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        for row in rows:
            csv_writer.writerow(row)


async def do_action(request: web.Request) -> web.Response:
    endlog.debug("Init action request")
    request_data = await request.json()
    name = request_data.get("name")
    args = request_data.get("args")

    endlog.debug(f"Received action request: {name} {args}")

    method_name = f"ACTIONS_{name}"
    method = globals().get(method_name)

    if callable(method):
        result = None
        if args:
            result = method(*args) if isinstance(args, list) else method(args)
        else:
            result = method()

        endlog.debug(f"Action result: {result}")
        return web.json_response({"result": result})

    available_methods = [
        attr[len("ACTIONS_") :]
        for attr in globals()
        if attr.startswith("ACTIONS_")
    ]

    return web.json_response(
        {
            "error": "Invalid method name.",
            "available_methods": available_methods,
        }
    )


# - HTML UTILS


def dependencies_button(name: str, dependencies: list[str]) -> str:
    deps = ",".join([f"'{x}'" for x in dependencies])
    return f"""
        <button
            class="dependency-button"
            onclick="window.mtb_action('installDependency',[{deps}])"
        >Install {name} deps</button>
        """


def csv_editor():
    inputs = [f for f in styles_dir.iterdir() if f.suffix == ".csv"]
    # rows = {f.stem: list(csv.reader(f.read_text("utf8"))) for f in styles}

    style_files = {}
    for file in inputs:
        with open(file, encoding="utf8") as f:
            parsed = csv.reader(f)
            style_files[file.name] = []
            for row in parsed:
                endlog.debug(f"Adding style {row[0]}")
                style_files[file.name].append((row[0], row[1], row[2]))

    html_out = """
            <div id="style-editor">
             <h1>Style Editor</h1>

            """
    for current, styles in style_files.items():
        current_out = f"<h3>{current}</h3>"
        table_rows = []
        for index, style in enumerate(styles):
            table_rows += (
                (["<tr>"] + [f"<th>{cell}</th>" for cell in style] + ["</tr>"])
                if index == 0
                else (
                    ["<tr>"]
                    + [
                        f"<td><input type='text' value='{cell}'></td>"
                        if i == 0
                        else f"<td><textarea name='Text1' cols='40' rows='5'>{cell}</textarea></td>"
                        for i, cell in enumerate(style)
                    ]
                    + ["</tr>"]
                )
            )
        current_out += (
            f"<table data-id='{current}' data-filename='{current}'>"
            + "".join(table_rows)
            + "</table>"
        )
        current_out += f"<button data-id='{current}' onclick='saveTableData(this.getAttribute(\"data-id\"))'>Save {current}</button>"

        html_out += add_foldable_region(current, current_out)

    html_out += "</div>"
    html_out += """<script src='/mtb-assets/js/saveTableData.js'></script>"""

    return html_out


def render_tab_view(**kwargs):
    tab_headers = []
    tab_contents = []

    for idx, (tab_name, content) in enumerate(kwargs.items()):
        active_class = "active" if idx == 0 else ""
        tab_headers.append(
            f"<button class='tablinks {active_class}' onclick=\"openTab(event, '{tab_name}')\">{tab_name}</button>"
        )
        tab_contents.append(
            f"<div id='{tab_name}' class='tabcontent {active_class}'>{content}</div>"
        )

    headers_str = "\n".join(tab_headers)
    contents_str = "\n".join(tab_contents)

    return f"""
<div class='tab-container'>
    <div class='tab'>
        {headers_str}
    </div>
    {contents_str}
    </div>
    <script src='/mtb-assets/js/tabSwitch.js'></script>
    """


def add_foldable_region(title: str, content: str):
    symbol_id = f"{title}-symbol"
    return f"""
    <div class='foldable'>
        <div
            class='foldable-title'
            onclick="toggleFoldable('{title}', '{symbol_id}')"
        >
            <span id='{symbol_id}' class='foldable-symbol'>&#9655;</span>
            {title}
        </div>
        <div id='{title}' class='foldable-content'>
            {content}
        </div>
    </div>
    <script src='/mtb-assets/js/foldable.js'></script>
    """


def add_split_pane(
    left_content: str, right_content: str, *, vertical: bool = True
):
    orientation = "vertical" if vertical else "horizontal"
    return f"""
    <div class="split-pane {orientation}">
        <div id="leftPane">
            {left_content}
        </div>
        <div id="resizer"></div>
        <div id="rightPane">
            {right_content}
        </div>
    </div>
    <script>
        initSplitPane({str(vertical).lower()});
    </script>
    <script src='/mtb-assets/js/splitPane.js'></script>
    """


def add_dropdown(title, options):
    option_str = "\n".join(
        [f"<option value='{opt}'>{opt}</option>" for opt in options]
    )
    return f"""
    <select>
        <option disabled selected>{title}</option>
        {option_str}
    </select>
    """


def render_table(table_dict: dict[str, Any], sort=True, title=None):
    table_list = sorted(
        table_dict.items(), key=lambda item: item[0]
    )  # Sort the dictionary by keys

    table_rows = ""
    for name, item in table_list:
        if isinstance(item, dict):
            if "dependencies" in item:
                table_rows += f"<tr><td>{name}</td><td>"
                table_rows += (
                    f"{dependencies_button(name,item['dependencies'])}"
                )

                table_rows += "</td></tr>"
            else:
                table_rows += (
                    f"<tr><td>{name}</td><td>{render_table(item)}</td></tr>"
                )
        # elif isinstance(item, str):
        #     table_rows += f"<tr><td>{name}</td><td>{item}</td></tr>"
        else:
            table_rows += f"<tr><td>{name}</td><td>{item}</td></tr>"

    return f"""
        <div class="table-container">
        {"" if title is None else f"<h1>{title}</h1>"}
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        </div>
        """


def render_base_template(title: str, content: str):
    github_icon_svg = """<svg xmlns="http://www.w3.org/2000/svg" fill="whitesmoke" height="3em" viewBox="0 0 496 512"><path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg>"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <link rel="stylesheet" href="/mtb-assets/style.css"/>
    </head>
    <script type="module">
        import {{ api }} from '/scripts/api.js'
        const mtb_action = async (action, args) =>{{
            console.log(`Sending ${{action}} with args: ${{args}}`)
            }}
        window.mtb_action = async (action, args) =>{{
            console.log(`Sending ${{action}} with args: ${{args}} to the API`)
            const res = await api.fetchApi('/actions', {{
                method: 'POST',
                body: JSON.stringify({{
                  name: action,
                  args,
                }}),
            }})

              const output = await res.json()
              console.debug(`Received ${{action}} response:`, output)
              if (output?.result?.error){{
                  alert(`An error occured: {{output?.result?.error}}`)
              }}
              return output?.result
        }}
    </script>
    <body>
        <header>
        <a href="/">Back to Comfy</a>
        <div class="mtb_logo">
            <img
                src="https://repository-images.githubusercontent.com/649047066/a3eef9a7-20dd-4ef9-b839-884502d4e873"
                alt="Comfy MTB Logo" height="70" width="128">
            <span class="title">Comfy MTB</span></div>
            <a style="width:128px;text-align:center" href="https://www.github.com/melmass/comfy_mtb">
                {github_icon_svg}
            </a>
        </header>

        <main>
            {content}
        </main>

        <footer>
            <!-- Shared footer content here -->
        </footer>
    </body>

    </html>
    """
