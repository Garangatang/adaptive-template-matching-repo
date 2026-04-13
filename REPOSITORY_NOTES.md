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

## Suggested Next Steps

1. `cd adaptive-template-matching-repo`
2. `git init`
3. Update the placeholder GitHub URLs in `pyproject.toml`
4. `pip install -e .`
5. Commit and push to GitHub
