# Adaptive Template Matching

Installable Python package for the adaptive template matching workflow in `Adaptive_Template_Matching.py`, plus the manual inflection-point marking GUI from `GT_GUI_Parsing/Manual_Inflection_Point_Marking_GUI.py`.

## Install

```bash
pip install .
```

For development from a cloned GitHub repository:

```bash
pip install -e .
```

Qt GUI dependency selection is platform-aware during installation:

- macOS and non-Windows platforms install `PyQt5`
- Windows installs `PySide6`

## Included Components

- `adaptive_template_matching.AdaptiveTemplateMatching`: adaptive template matcher for gait signal processing.
- `atm-manual-inflection-gui`: console entry point for the manual inflection-point marking GUI.

## Usage

```python
from adaptive_template_matching import AdaptiveTemplateMatching

matcher = AdaptiveTemplateMatching()
```

Launch the GUI after installation:

```bash
atm-manual-inflection-gui
```

## Repository Layout

```text
src/adaptive_template_matching/
├── __init__.py
├── matcher.py
└── gui/
    ├── __init__.py
    └── manual_inflection_point_marking_gui.py
```

## Notes

- The package dependencies in `pyproject.toml` include everything needed for both the matcher and the GUI.
- The GUI chooses `PyQt5` on macOS, `PySide6` on Windows, and `PyQt5` as the fallback elsewhere.
- The original source files are still present in the repository root for reference, while the installable package lives under `src/`.
