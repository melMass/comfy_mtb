import os
import ast
import json
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console(stderr=True)


def get_imported_modules(filename):
    with open(filename, "r") as file:
        tree = ast.parse(file.read())

    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(
                (alias.name, alias.name in sys.builtin_module_names)
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(
                    (node.module, node.module in sys.builtin_module_names)
                )

    return imported_modules


def list_imported_modules(folder):
    modules = []

    file_count = sum(len(files) for _, _, files in os.walk(folder))
    progress = Progress()

    task = progress.add_task("[cyan]Scanning files...", total=file_count)

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                imported_modules = get_imported_modules(file_path)
                modules.extend(imported_modules)
            progress.update(task, advance=1)

    progress.stop()

    return modules


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print(
            "[bold red]Please provide the folder path as a command-line argument.[/bold red]"
        )
        sys.exit(1)

    # folder_path = input("Enter the folder path: ")
    # while not os.path.isdir(folder_path):
    #     console.print("[bold red]Invalid folder path![/bold red]")
    #     folder_path = input("Enter the folder path: ")
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        console.print("[bold red]Invalid folder path![/bold red]")
        sys.exit(1)

    console.print("[bold green]=== Python Imported Modules ===[/bold green]\n")
    console.print(f"Scanning folder: [bold]{folder_path}[/bold]\n")

    imported_modules = list_imported_modules(folder_path)

    console.print(f"\n[bold green]Imported Modules:[/bold green]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Module")
    table.add_column("Type")

    for module, is_builtin in imported_modules:
        module_type = "Built-in" if is_builtin else "External"
        table.add_row(module, module_type)

    console.print(table)

    json_data = json.dumps(
        [
            {"module": module, "type": "Built-in" if is_builtin else "External"}
            for module, is_builtin in imported_modules
        ],
        indent=4,
    )
    print(json_data)
