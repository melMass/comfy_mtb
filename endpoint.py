from .utils import here, run_command, comfy_mode, import_install
from aiohttp import web
from .log import mklog
import sys

endlog = mklog("mtb endpoint")

# - ACTIONS
import platform

import_install("requirements")


def ACTIONS_installDependency(dependency_names=None):
    if dependency_names is None:
        return {"error": "No dependency name provided"}
    endlog.debug(f"Received Install Dependency request for {dependency_names}")
    reqs = []
    if platform.system() == "Windows":
        reqs = list(requirements.parse((here / "reqs_windows.txt").read_text()))
    else:
        reqs = list(requirements.parse((here / "reqs.txt").read_text()))
    print([x.specs for x in reqs])
    print(
        "\n".join([f"{x.line} {''.join(x.specs[0] if x.specs else '')}" for x in reqs])
    )
    for dependency_name in dependency_names:
        for req in reqs:
            if req.name == dependency_name:
                endlog.debug(f"Dependency {dependency_name} installed")
                break
    return {"success": True}


def ACTIONS_getStyles(style_name=None):
    from .nodes.conditions import StylesLoader

    styles = StylesLoader.options
    match_list = ["name"]
    if styles:
        filtered_styles = {
            key: value
            for key, value in styles.items()
            if not key.startswith("__") and key not in match_list
        }
        if style_name:
            return filtered_styles.get(style_name, {"error": "Style not found"})
        return filtered_styles
    return {"error": "No styles found"}


async def do_action(request) -> web.Response:
    endlog.debug("Init action request")
    request_data = await request.json()
    name = request_data.get("name")
    args = request_data.get("args")

    endlog.debug(f"Received action request: {name} {args}")

    method_name = f"ACTIONS_{name}"
    method = globals().get(method_name)

    if callable(method):
        result = method(args) if args else method()
        endlog.debug(f"Action result: {result}")
        return web.json_response({"result": result})

    available_methods = [
        attr[len("ACTIONS_") :] for attr in globals() if attr.startswith("ACTIONS_")
    ]

    return web.json_response(
        {"error": "Invalid method name.", "available_methods": available_methods}
    )


# - HTML UTILS


def dependencies_button(name, dependencies):
    deps = ",".join([f"'{x}'" for x in dependencies])
    return f"""
        <button class="dependency-button" onclick="window.mtb_action('installDependency',[{deps}])">Install {name} deps</button>
        """


def render_table(table_dict, sort=True, title=None):
    table_dict = sorted(
        table_dict.items(), key=lambda item: item[0]
    )  # Sort the dictionary by keys

    table_rows = ""
    for name, item in table_dict:
        if isinstance(item, dict):
            if "dependencies" in item:
                table_rows += f"<tr><td>{name}</td><td>"
                table_rows += f"{dependencies_button(name,item['dependencies'])}"

                table_rows += "</td></tr>"
            else:
                table_rows += f"<tr><td>{name}</td><td>{render_table(item)}</td></tr>"
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


def render_base_template(title, content):
    css_content = ""
    css_path = here / "html" / "style.css"
    if css_path:
        with open(css_path, "r") as css_file:
            css_content = css_file.read()

    github_icon_svg = """<svg xmlns="http://www.w3.org/2000/svg" fill="whitesmoke" height="3em" viewBox="0 0 496 512"><path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg>"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            {css_content}
        </style>
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
            <img src="https://repository-images.githubusercontent.com/649047066/a3eef9a7-20dd-4ef9-b839-884502d4e873" alt="Comfy MTB Logo" height="70" width="128">
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
