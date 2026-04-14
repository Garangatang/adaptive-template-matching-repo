# Repository Notes

This directory is a standalone, GitHub-ready repository layout for the adaptive template matching project.

## Contents

- `src/adaptive_template_matching/`: installable Python package
- `pyproject.toml`: packaging metadata and dependencies
- `README.md`: install and usage overview
- `legacy_sources/`: copies of the original source files for reference

## Dependency Notes

- `pyproject.toml` uses platform markers for the GUI toolkit.
- Windows installs `PySide6`.
- macOS and other non-Windows platforms install `PyQt5`.

