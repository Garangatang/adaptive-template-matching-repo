# Adaptive Template Matching

Installable Python package for the adaptive template matching workflow in `Adaptive_Template_Matching.py`, plus the manual inflection-point marking GUI from `GT_GUI_Parsing/Manual_Inflection_Point_Marking_GUI.py`.

## What is it

A Python program which utilizes an adaptive template to mark all heel strike and toe off occurences in ground reaction force (GRF) data.

The template also can be adapted to other sensor signals, but has not been fully tested on other sensor signals besides GRF data at this time.

## Video of How Automated Template Matching Works

<video src= "https://github.com/user-attachments/assets/4c19e08e-cad9-4a7a-853f-686858dec092" width="320" height="240" controls></video>

## Install

For development from a cloned GitHub repository:

```bash
pip install git+https://github.com/Garangatang/adaptive-template-matching-repo
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

## License and Citation

This repository is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Use, sharing, adaptation, or redistribution of this work requires attribution to Adaptive Template Matching, Grange Simpson, and Ivan Khimach, plus citation of the associated academic paper:

```text
[Academic paper citation and link to be added.]
```

Until the paper citation is added, cite the project repository and authors and retain the citation placeholder in redistributions.

## Previous Versions

A non-automated V1 version is located [here](https://github.com/Garangatang/Non_Automated_Adaptive_Gait_Cycle_Template_Matching/tree/main)

## Collaborators

[![Github Badge](https://img.shields.io/badge/-Grange_Simpson-3A3B3C?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Garangatang)
[![Github Badge](https://img.shields.io/badge/-Ivan_Khimach-3A3B3C?style=for-the-badge&logo=github&logoColor=white)](https://github.com/KiViKatt)
[![Github Badge](https://img.shields.io/badge/-NERVES_Lab-3A3B3C?style=for-the-badge&logo=github&logoColor=white)](https://github.com/NERVESLabUtah)
