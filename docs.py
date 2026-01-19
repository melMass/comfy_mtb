###
# File: docs.py
# Project: comfy_mtb
# Author: Mel Massadian
# Copyright (c) 2023-2025 Mel Massadian
#
###
"""Documentation handling utilities for MTB nodes."""

import re
from pathlib import Path


def strip_html_tags(text: str) -> str:
    """Strip HTML tags from description text, converting images to markdown.

    - Removes <details>...</details> blocks entirely (they contain JSON workflows)
    - Converts <img src="URL"> to ![](URL)
    - Removes all other HTML tags while keeping their content
    """
    # Remove <details>...</details> blocks entirely (contain JSON workflows)
    text = re.sub(r"<details>.*?</details>", "", text, flags=re.DOTALL)
    # Convert <img src="URL"> to ![](URL)
    text = re.sub(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*/?\s*>', r"![](\1)", text)
    # Remove remaining HTML tags, keep content
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def wiki_to_classname(s: str) -> str:
    """Convert wiki filename to class name.

    Example: "nodes-animation-builder" -> "MTB_AnimationBuilder"
    """
    wiki_name = s.replace("nodes-", "", 1)
    return "MTB_" + "".join([part.capitalize() for part in wiki_name.split("-")])


def classname_to_wiki(s: str) -> str:
    """Convert class name to wiki filename.

    Example: "MTB_AnimationBuilder" -> "nodes-animation-builder"
    """
    classname = s.replace("MTB_", "")
    parts: list[str] = []
    start = 0
    for i in range(1, len(classname)):
        if classname[i].isupper():
            parts.append(classname[start:i].lower())
            start = i
    parts.append(classname[start:].lower())
    return "nodes-" + "-".join(parts)


def load_wiki_docs(wiki_path: Path) -> dict[str, str]:
    """Load wiki documentation files as a dict of classname -> content."""
    if not wiki_path.exists() or not wiki_path.is_dir():
        return {}
    nodes_path = wiki_path / "nodes"
    if not nodes_path.exists():
        return {}
    return {
        wiki_to_classname(x.stem): x.read_text(encoding="utf-8")
        for x in nodes_path.glob("*.md")
    }


def assign_descriptions(
    nodes: list,
    node_docs: dict[str, str],
    wiki_path: Path,
    log,
    export: bool = False,
) -> None:
    """Assign DESCRIPTION to nodes from wiki docs or docstrings.

    Priority:
    1. Existing DESCRIPTION attribute (not modified)
    2. Wiki doc file
    3. __doc__ docstring
    """
    for node_class in nodes:
        class_name = node_class.__name__
        linked_doc = node_docs.get(class_name)

        if not hasattr(node_class, "DESCRIPTION"):
            if linked_doc:
                log.debug(f"Found linked doc for {class_name}, using it")
                node_class.DESCRIPTION = linked_doc
            elif node_class.__doc__:
                log.debug(f"Using __doc__ as description for {class_name}")
                node_class.DESCRIPTION = node_class.__doc__
                if export:
                    wiki_name = classname_to_wiki(class_name)
                    (wiki_path / "nodes" / f"{wiki_name}.md").write_text(
                        node_class.__doc__, encoding="utf-8"
                    )
            else:
                log.debug(
                    f"None of the methods could retrieve documentation for {class_name}"
                )
