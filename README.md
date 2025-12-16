# FMRPreview

Lightweight PyQt5 + pyqtgraph viewer for CSV-based FMR data with optional derivative/integral overlays and phase/sweep corrections.

## Features
- Auto-detects newest CSV in `..\Data` on launch and parses numeric columns with units.
- Plot multiple files (up to the configured limit), per-trace scaling/weight/color, and derivative/integral overlay.
- Phase rotation for X/Y, sweep-direction correction, and quick phase guessing.
- Configurable defaults in a sidecar ASCII JSON config (`FMRPreview.cfg`).

## Setup
1) Install Python 3.13 (64-bit recommended) and create/activate a venv.  
2) Install deps:
   ```
   pip install pyqt5 pyqtgraph numpy
   ```
3) Adjust `FMRPreview.cfg` (data path, colors, limits). Default data path is `..\\Data`; change it to your normal data location. If missing, the app recreates the file with defaults.

## Run
```
python FMRPreview.py
```
The UI starts maximized, auto-loads the newest CSV under `Data/YYYYMMDD` if present, and shows header/axes selectors and plot controls.

## Config
- File: `FMRPreview.cfg` (JSON, ASCII).  
- Keys: `DATA_DIR_PATH`, `SWEEP_ERROR_COEFFICIENT`, `IGNORE_DIRECTION_POINTS`, `PHASE_CUTOFF_FIELD`, `MAX_FILES`, `DEFAULT_COLORS`.
- The app writes a default file if one is not found next to `FMRPreview.py`.

## Build (Nuitka)
Requires MSVC Build Tools 2022 (C++ toolchain + Windows 10/11 SDK) and Nuitka:
```
pip install -U nuitka zstandard
```
Use the helper script (folder layout):
```
./build_nuitka.ps1
```
Single-file exe:
```
./build_nuitka.ps1 -OneFile
```
Include `Data/` contents:
```
./build_nuitka.ps1 -IncludeDataDir
```
Notes:
- PyQt5 support is considered legacy by Nuitka; PySide6/Qt6 is better supported if you switch bindings.
- To trim unused optional deps (e.g., matplotlib/scipy pulled in by pyqtgraph), add `--noinclude-module=matplotlib.*` and `--noinclude-module=scipy.*` to the script and retest.

## Repository hygiene
- `.gitignore` excludes Python bytecode, Nuitka build outputs (`*.build/`, `*.dist/`, `.nuitka-cache/`, onefile artifacts), and common temp/log files.
