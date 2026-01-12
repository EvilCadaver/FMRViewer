# FMRPreview (FMRViewer)

Lightweight PyQt5 + pyqtgraph viewer for CSV-based FMR data with optional derivative/integral overlays and phase/sweep corrections.

## Quick start
1) Install Python 3.13 (64-bit recommended) and create/activate a venv.
2) Install dependencies:
   ```
   pip install pyqt5 pyqtgraph numpy
   ```
3) Run:
   ```
   python FMRPreview.py
   ```

On launch, the UI maximizes, auto-loads the newest CSV under `..\Data` (if present), and exposes header/axis selectors and plot controls.

## Features
- Auto-detects the newest CSV in `..\Data` and parses numeric columns with units.
- Plot multiple files (up to the configured limit), per-trace scaling/weight/color.
- Derivative and integral overlays for each trace.
- Phase rotation for X/Y, sweep-direction correction, and quick phase guessing.

## Data layout
The default search path is `..\Data`, with optional `YYYYMMDD` subfolders. Update the path in the config if your data lives elsewhere.

## Configuration
Config file: `FMRPreview.cfg` (ASCII JSON) next to `FMRPreview.py`.

Key settings:
- `DATA_DIR_PATH`
- `SWEEP_ERROR_COEFFICIENT`
- `IGNORE_DIRECTION_POINTS`
- `PHASE_CUTOFF_FIELD`
- `MAX_FILES`
- `DEFAULT_COLORS`

If the file is missing, the app recreates it with defaults.

## Repository hygiene
`.gitignore` excludes Python bytecode and common temp/log files.
