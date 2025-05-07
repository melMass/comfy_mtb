# Contributing to mtb

Thank you for your interest in contributing to mtb!  We appreciate your help in making this project better.  This document outlines how you can contribute to the project.

## Project Overview

This project is a collection of custom nodes for ComfyUI, tailored specifically for animation workflows. It aims to provide a streamlined and user-friendly experience for creating animations within the ComfyUI environment.

## Ways to Contribute

We welcome all kinds of contributions! Here's how you can get involved:

*   **Bug Reports:**  If you encounter any issues, please create a new issue on GitHub.  Please include clear steps to reproduce the bug, along with any relevant error messages, workflows or screenshots.
*   **Feature Requests:**  Have an idea for a new node or feature?  Create a new issue to discuss it!  Please describe the feature in detail, and explain how it would benefit the project.
*   **Documentation Improvements:**  Help us improve the documentation by fixing errors, adding examples, or clarifying explanations.
*   **Code Contributions:**  We welcome contributions to the codebase!  Please see the "Development Setup" and "File Structure" sections below for more information.
*   **Testing:**  Help us ensure the stability and reliability of the project by testing new features and bug fixes.
*   **Refactoring:**  Help us improve the codebase by refactoring existing code to improve readability, maintainability, and performance.

## Development Setup

```sh
git clone --recursive https://github.com/melmass/comfy_mtb
```

## File Structure

Understanding the project structure is crucial for making effective contributions.

*   **`./nodes/*.py`:** This directory contains the definitions for all custom nodes. Nodes are automatically registered when a file defines an array named `__nodes__` containing the node classes.  Make sure your node follows the ComfyUI node definition structure.
*   **`./web/*.js`:** This directory contains all the frontend JavaScript code for the extension's user interface.
*   **`./wiki`:** This directory is a Git submodule that contains the project's Wiki documentation, written in Markdown.  Node documentation should be created or updated in the corresponding Markdown files within this submodule. This is then referenced by the UI for in-GUI help

## Coding Style

We use **Ruff** for code formatting to ensure consistency.  Please run Ruff on your code before submitting a pull request.  No specific configuration is required, so the default Ruff settings will be used.

## Contribution Workflow

1.  **Create a Branch:** Create a new branch for your feature or fix.  Use a descriptive branch name (e.g., `feature/new-node`, `fix/bug-in-ui`). **Do not fork the main branch directly.**
2.  **Make Changes:** Implement your changes in your branch.
3.  **Run Tests:** (Add instructions on how to run tests if available.)
4.  **Format Code:** Run Ruff on your code to ensure it is properly formatted.
5.  **Create a Pull Request:** Submit a pull request to the `main` branch.  Please provide a clear and concise description of your changes.

## Code of Conduct

We are committed to creating a welcoming and inclusive community.  We expect all contributors to adhere to a respectful and professional code of conduct. (Consider adding a link to a CODE_OF_CONDUCT.md file or a standard code of conduct.)

## Tools and Libraries

*   **Python:** The primary programming language for this project.
*   **ComfyUI:** The underlying framework for the custom nodes.

## Current Focus

We are currently focused on a major refactor to clean up the project's codebase. Contributions related to this effort are particularly welcome!

## Thank You!

Thank you for considering contributing to mtb! Your contributions are greatly appreciated.  We look forward to reviewing your pull requests!

