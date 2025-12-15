import csv
import glob
import math
import os
import sys
from functools import partial
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np

# Lightweight CSV plotting helper with optional derivative/integral overlay

DATA_DIR_NAME = "Data"
SWEEP_ERROR_COEFFICIENT = 0.38
IGNORE_DIRECTION_POINTS = 10  # points at start assumed positive sweep
PHASE_CUTOFF_FIELD = 500  # Oe to estimate initial phase
MAX_FILES = 16
DEFAULT_COLORS = [
    (220, 70, 70),
    (60, 130, 220),
    (60, 170, 140),
    (220, 150, 60),
    (140, 80, 220),
    (0, 170, 200),
    (200, 110, 110),
    (110, 200, 110),
    (210, 210, 80),
    (110, 110, 210),
    (240, 130, 40),
    (40, 160, 240),
    (160, 200, 60),
    (200, 80, 170),
    (120, 120, 120),
    (60, 60, 60),
]


def parse_numeric_columns(csv_path: str):
    """
    Parse a CSV file that uses the pattern:
        header row: column names (e.g., Time,Field,X,Y,R,theta)
        units row : units for each column (e.g., s,Oe,V,V,V,deg)
        data rows : numeric values

    Returns two dicts and the detected file header text that precedes the column names:
        columns[name] -> list of float values
        units[name]   -> unit string (or empty if missing)
        header_text   -> lines before the CSV header (e.g., sample name, params)
    """
    with open(csv_path, newline="") as handle:
        raw_lines = [line.rstrip("\n") for line in handle]

    # Pick a header row. Prefer a row containing "field" to match typical data,
    # otherwise fall back to the first comma-separated row.
    header_index = None
    for idx, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line or "," not in line:
            continue
        if "field" in line.lower():
            header_index = idx
            break
        if header_index is None:
            header_index = idx
    if header_index is None:
        raise ValueError("Could not find a header row with comma-separated columns.")

    header_text_lines = [ln for ln in raw_lines[:header_index] if ln.strip()]

    # Build CSV-ready lines from the header forward, skipping empties
    lines = [line.strip() for line in raw_lines[header_index:] if line.strip()]

    # Create a CSV reader from the header forward; rows are simple lists at this point.
    reader = csv.reader(lines)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("File ended before reading header.")  # noqa: TRY200

    # Try to read a units row (must match header width). If it looks wrong, ignore it.
    units_row = next(reader, None)
    if units_row and len(units_row) != len(header):
        units_row = None

    data_rows = list(reader)  # Everything after header (+optional units) is data

    # Seed dicts with column names and (optional) units
    columns = {name.strip(): [] for name in header}
    units = {
        name.strip(): (units_row[idx].strip() if units_row and idx < len(units_row) else "")
        for idx, name in enumerate(header)
    }

    # Parse numeric rows; skip any row that fails float conversion
    for row in data_rows:
        if len(row) < len(header):
            continue
        try:
            values = [float(value) for value in row[: len(header)]]
        except ValueError:
            continue
        for name, value in zip(header, values):
            columns[name.strip()].append(value)

    # Drop columns that never produced numeric data and align units accordingly
    columns = {name: vals for name, vals in columns.items() if vals}
    units = {name: units.get(name, "") for name in columns}

    header_text = "\n".join(header_text_lines).strip()
    return columns, units, header_text


class PlotSession:
    """Container for a loaded CSV and its plot settings."""

    def __init__(self, path: str, columns: dict, units: dict, header_text: str, direction_data: list):
        self.path = path
        self.columns = columns
        self.units = units
        self.header_text = header_text
        self.direction_data = direction_data
        self.base_color = None  # color assigned from the bank
        self.color = None
        self.weight = 2
        self.scale = 1.0
        self.phase_deg = 0.0
        self.sweep_error = 0.0
        self.rotated_xy = None  # cached rotated X/Y for current phase


class FMRPreview(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FMR CSV Preview")
        # Default data directory lives alongside the script in Data/
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR_NAME)

        # In-memory state for loaded CSVs
        self.sessions = []  # list[PlotSession]
        self.selected_session_idx = -1
        self.color_queue = list(DEFAULT_COLORS)
        self.display_to_column = {}  # "Field (Oe)" -> "Field"
        self._suppress_combo = False  # guard to prevent signal loops while we adjust combos
        self._last_x_display = ""  # previous X selection (for swap logic)
        self._last_y_display = ""  # previous Y selection (for swap logic)
        self._secondary_curves = []  # plot items for derivative/integral overlay
        self._suppress_phase = False
        self.current_file_path = ""

        # PyQtGraph plot surface
        self.plot_widget = pg.PlotWidget(background="w")
        # Shared grid for primary axes; keep both directions visible
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "X (V)")
        self.plot_widget.setLabel("bottom", "Field (Oe)")
        # Attach a secondary ViewBox for the right-side axis
        plot_item = self.plot_widget.getPlotItem()
        plot_item.showAxis("right")
        right_axis = plot_item.getAxis("right")
        right_axis.setStyle(showValues=False)  # hide tick labels
        right_axis.setTicks([])  # hide tick marks
        right_axis.setPen(pg.mkPen(None))  # hide axis line
        right_axis.setTextPen(pg.mkPen(None))  # hide label pen
        self.secondary_vb = pg.ViewBox()
        plot_item.scene().addItem(self.secondary_vb)
        plot_item.getAxis("right").linkToView(self.secondary_vb)
        self.secondary_vb.setXLink(plot_item.vb)
        plot_item.vb.sigResized.connect(self.update_secondary_view_bounds)
        self.update_secondary_view_bounds()

        # File path (selectable) + button to pick a CSV
        self.file_path_edit = QtWidgets.QLineEdit("No file loaded")
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setMinimumWidth(300)
        self.open_button = QtWidgets.QPushButton("Open CSV...")
        self.open_button.clicked.connect(self.choose_file)
        self.last_viewed_dir = None  # remember last folder visited during this session

        self.header_display = QtWidgets.QPlainTextEdit()
        self.header_display.setReadOnly(True)
        self.header_display.setPlaceholderText("Header will show after loading a file.")
        self.header_display.setMaximumHeight(90)
        self.header_display.setMinimumHeight(60)
        self.header_display.setPlainText("Header will show after loading a file.")

        # Axis selectors. Each combo shows column names (with units when available).
        self.x_combo = QtWidgets.QComboBox()
        self.y_combo = QtWidgets.QComboBox()
        self.x_combo.currentTextChanged.connect(self.on_x_changed)
        self.y_combo.currentTextChanged.connect(self.on_y_changed)

        axes_row = QtWidgets.QHBoxLayout()
        axes_row.addWidget(QtWidgets.QLabel("Abscissa (X):"))
        axes_row.addWidget(self.x_combo)
        axes_row.addSpacing(12)
        axes_row.addWidget(QtWidgets.QLabel("Ordinate (Y):"))
        axes_row.addWidget(self.y_combo)
        axes_row.addStretch()

        # Secondary calculations (derivative/integral) controls
        self.derivative_checkbox = QtWidgets.QCheckBox("Show derivative (dY/dX)")
        self.integral_checkbox = QtWidgets.QCheckBox("Show integral (Y*dX)")
        self.derivative_checkbox.stateChanged.connect(self.on_derivative_toggled)
        self.integral_checkbox.stateChanged.connect(self.on_integral_toggled)

        # Phase rotation controls for X+iY
        self.phase_dial = QtWidgets.QDial()
        self.phase_dial.setRange(-180, 180)
        self.phase_dial.setNotchesVisible(True)
        self.phase_dial.setMinimumSize(150, 150)  # make the wheel visibly larger
        self.phase_dial.valueChanged.connect(self.on_phase_dial_changed)

        self.phase_plus_button = QtWidgets.QPushButton("+90°")
        self.phase_plus_button.clicked.connect(self.on_phase_plus_90)
        self.phase_minus_button = QtWidgets.QPushButton("-90°")
        self.phase_minus_button.clicked.connect(self.on_phase_minus_90)

        self.phase_spin = QtWidgets.QDoubleSpinBox()
        self.phase_spin.setDecimals(2)
        self.phase_spin.setRange(-180.0, 180.0)
        self.phase_spin.setSingleStep(1.0)
        self.phase_spin.valueChanged.connect(self.on_phase_spin_changed)
        self.set_phase(0.0, update=False)

        self.guess_phase_button = QtWidgets.QPushButton("Guess phase")
        self.guess_phase_button.clicked.connect(self.on_guess_phase)
        self.reset_phase_button = QtWidgets.QPushButton("Reset phase")
        self.reset_phase_button.clicked.connect(self.on_reset_phase)

        phase_spin_row = QtWidgets.QHBoxLayout()
        phase_spin_row.addWidget(QtWidgets.QLabel("Phase shift:"))
        phase_spin_row.addWidget(self.phase_spin)
        phase_spin_row.addWidget(self.guess_phase_button)
        phase_spin_row.addWidget(self.reset_phase_button)
        
        dial_buttons_col = QtWidgets.QVBoxLayout()
        dial_buttons_col.addWidget(self.phase_plus_button)
        dial_buttons_col.addSpacing(6)
        dial_buttons_col.addWidget(self.phase_minus_button)
        
        phase_dial_row = QtWidgets.QHBoxLayout()
        phase_dial_row.addWidget(self.phase_dial)
        phase_dial_row.addSpacing(6)
        phase_dial_row.addLayout(dial_buttons_col)
        phase_dial_row.addStretch()

        # Sweep error correction control (suggested from dField/dt near 25%)
        self.sweep_error_spin = QtWidgets.QDoubleSpinBox()
        self.sweep_error_spin.setDecimals(0)
        self.sweep_error_spin.setRange(0, 1000)
        self.sweep_error_spin.setSingleStep(1)
        self.sweep_error_spin.valueChanged.connect(self.on_sweep_error_changed)

        sweep_row = QtWidgets.QHBoxLayout()
        sweep_row.addWidget(QtWidgets.QLabel("Sweep error shift:"))
        sweep_row.addWidget(self.sweep_error_spin)
        sweep_row.addWidget(QtWidgets.QLabel("(Field - Direction * error)"))
        sweep_row.addStretch()

        options_row = QtWidgets.QHBoxLayout()
        options_row.addWidget(self.derivative_checkbox)
        options_row.addSpacing(12)
        options_row.addWidget(self.integral_checkbox)
        options_row.addStretch()

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.file_path_edit, 1)
        top_row.addWidget(self.open_button)

        header_layout = QtWidgets.QVBoxLayout()
        # header_layout.addWidget(QtWidgets.QLabel("File header:"))
        header_layout.addWidget(self.header_display)

        self.files_list_layout = QtWidgets.QVBoxLayout()
        self.files_list_layout.setContentsMargins(0, 0, 0, 0)
        self.files_list_layout.setSpacing(4)
        self.files_list_layout.addStretch()
        files_column = QtWidgets.QVBoxLayout()
        files_column.addWidget(QtWidgets.QLabel("Loaded files (max " + str(MAX_FILES) + "):"))
        files_column.addLayout(self.files_list_layout, 1)
        files_column.addStretch()

        right_col = QtWidgets.QVBoxLayout()
        right_col.addLayout(header_layout)
        right_col.addLayout(axes_row)
        right_col.addLayout(options_row)
        right_col.addLayout(sweep_row)
        right_col.addLayout(phase_spin_row)
        right_col.addLayout(phase_dial_row)
        right_col.addLayout(files_column)
        right_col.addStretch()

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.addLayout(right_col)
        controls_row.addSpacing(12)
        controls_row.addStretch()

        body_row = QtWidgets.QHBoxLayout()
        body_row.addWidget(self.plot_widget, 1)
        body_row.addLayout(controls_row)
        # body_row.addLayout(files_column)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(body_row)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_first_csv_if_present()

    def choose_file(self):
        """Open a file dialog and load one or more CSVs."""
        start_dir = (
            self.last_viewed_dir
            or self.find_latest_data_folder()
            or (self.data_dir if os.path.isdir(self.data_dir) else os.getcwd())
        )
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select CSV files",
            start_dir,
            "CSV files (*.csv);;All files (*.*)",
        )
        if paths:
            self.add_sessions_from_paths(paths)

    def find_latest_data_folder(self):
        """Return newest YYYYMMDD subfolder inside Data/, or None if absent."""
        if not os.path.isdir(self.data_dir):
            return None
        dated_dirs = [
            entry.path
            for entry in os.scandir(self.data_dir)
            if entry.is_dir() and entry.name.isdigit() and len(entry.name) == 8
        ]
        if not dated_dirs:
            return None
        # Names are YYYYMMDD so lexical sort matches chronological order
        return sorted(dated_dirs, reverse=True)[0]

    @staticmethod
    def find_newest_csv(directory: str):
        """Return newest .csv file in directory (by mtime), or None if missing."""
        if not directory or not os.path.isdir(directory):
            return None
        csv_files = [path for path in glob.glob(os.path.join(directory, "*.csv")) if os.path.isfile(path)]
        if not csv_files:
            return None
        return max(csv_files, key=os.path.getmtime)

    def load_first_csv_if_present(self):
        """Auto-load newest CSV from newest dated folder to show data immediately."""
        latest_folder = self.find_latest_data_folder()
        search_dir = latest_folder or (self.data_dir if os.path.isdir(self.data_dir) else None)
        newest_csv = self.find_newest_csv(search_dir) if search_dir else None
        if newest_csv:
            self.add_sessions_from_paths([newest_csv])

    def add_sessions_from_paths(self, paths):
        """Load one or more files into sessions, obeying the max limit."""
        if not paths:
            return
        existing_paths = {session.path for session in self.sessions}
        added = 0
        target_index = -1
        for path in paths:
            if len(self.sessions) >= MAX_FILES:
                QtWidgets.QMessageBox.information(
                    self, "Limit reached", f"Maximum of {MAX_FILES} files. Close some to open more."
                )
                break
            if not path or not os.path.isfile(path):
                continue
            if path in existing_paths:
                idx = next((i for i, sess in enumerate(self.sessions) if sess.path == path), -1)
                QtWidgets.QMessageBox.information(
                    self, "Already loaded", f"{os.path.basename(path)} is already loaded."
                )
                if idx >= 0:
                    target_index = idx
                continue  # already loaded
            session = self.build_session(path)
            if session is None:
                continue
            if session.color is None:
                if self.color_queue:
                    base_color = self.color_queue.pop(0)
                else:
                    base_color = DEFAULT_COLORS[len(self.sessions) % len(DEFAULT_COLORS)]
                session.base_color = base_color
                session.color = base_color
            self.sessions.append(session)
            added += 1
            existing_paths.add(path)
            self.last_viewed_dir = os.path.dirname(path)
            target_index = len(self.sessions) - 1

        if added:
            self.populate_axis_choices()
        if target_index != -1:
            self.select_session(target_index)
        elif added:
            self.update_file_list_ui()
            self.update_plot()

    def build_session(self, path: str):
        """Parse a file and create a PlotSession instance."""
        try:
            columns, units, header_text = parse_numeric_columns(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Load error", f"{path}:\n{exc}")
            return None
        if not columns:
            QtWidgets.QMessageBox.information(self, "No data", f"No numeric columns found in {path}.")
            return None
        direction_data, suggested_error = self.compute_direction_and_error(columns)
        # Attach direction column for plotting convenience
        if direction_data:
            columns = dict(columns)
            units = dict(units)
            columns["Direction"] = direction_data
            units["Direction"] = ""

        session = PlotSession(path, columns, units, header_text, direction_data)
        session.suggested_error = suggested_error
        session.sweep_error = suggested_error
        return session

    def update_file_list_ui(self):
        """Refresh the vertical list of loaded files and their controls."""
        while self.files_list_layout.count():
            item = self.files_list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        for idx, session in enumerate(self.sessions):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_widget.setLayout(row_layout)

            color_label = QtWidgets.QLabel()
            color_label.setFixedSize(14, 14)
            color = session.color or DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
            color_label.setStyleSheet(f"background-color: rgb{color}; border: 1px solid #444;")

            name_label = QtWidgets.QLabel(os.path.basename(session.path))
            name_label.setToolTip(session.path)
            name_label.mousePressEvent = lambda event, i=idx: self.select_session(i)

            adjust_btn = QtWidgets.QPushButton("Adjust…")
            adjust_btn.setMaximumWidth(80)
            adjust_btn.clicked.connect(partial(self.open_adjust_dialog, idx))

            scale_label = QtWidgets.QLabel(f"x{session.scale:g}")
            scale_label.setMinimumWidth(50)

            remove_btn = QtWidgets.QPushButton("X")
            remove_btn.setMaximumWidth(30)
            remove_btn.clicked.connect(partial(self.remove_session, idx))

            row_layout.addWidget(color_label)
            row_layout.addWidget(name_label, 1)
            row_layout.addWidget(adjust_btn)
            row_layout.addWidget(scale_label)
            row_layout.addWidget(remove_btn)

            if idx == self.selected_session_idx:
                row_widget.setStyleSheet("background-color: #dfe8ff;")
                font = name_label.font()
                font.setBold(True)
                name_label.setFont(font)
            else:
                row_widget.setStyleSheet("")
                font = name_label.font()
                font.setBold(False)
                name_label.setFont(font)

            self.files_list_layout.addWidget(row_widget)
        self.files_list_layout.addStretch()

    def remove_session(self, index: int):
        if index < 0 or index >= len(self.sessions):
            return
        removed = self.sessions.pop(index)
        if removed and removed.base_color:
            self.color_queue.append(removed.base_color)
        self.populate_axis_choices()
        if self.sessions:
            new_index = min(index, len(self.sessions) - 1)
            self.select_session(new_index)
        else:
            self.select_session(-1)

    def open_adjust_dialog(self, index: int):
        """Dialog to tweak color, weight, and scale for a session."""
        if index < 0 or index >= len(self.sessions):
            return
        self.select_session(index)
        session = self.sessions[index]
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Adjust plot: {os.path.basename(session.path)}")
        layout = QtWidgets.QFormLayout(dialog)

        color_value = session.color or DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
        color_preview = QtWidgets.QLabel()
        color_preview.setFixedSize(60, 18)
        color_preview.setStyleSheet(f"background-color: rgb{color_value}; border: 1px solid #444;")

        def pick_color():
            nonlocal color_value
            initial = QtGui.QColor(*color_value)
            chosen = QtWidgets.QColorDialog.getColor(initial, self, "Choose line color")
            if chosen.isValid():
                color_value = (chosen.red(), chosen.green(), chosen.blue())
                color_preview.setStyleSheet(
                    f"background-color: rgb{color_value}; border: 1px solid #444;"
                )

        color_btn = QtWidgets.QPushButton("Pick color")
        color_btn.clicked.connect(pick_color)
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(color_btn)
        color_row.addWidget(color_preview)
        color_row.addStretch()

        weight_spin = QtWidgets.QSpinBox()
        weight_spin.setRange(1, 10)
        weight_spin.setValue(session.weight)

        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setDecimals(4)
        scale_spin.setRange(0.0001, 1e6)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(session.scale)

        layout.addRow("Color:", color_row)
        layout.addRow("Line weight:", weight_spin)
        layout.addRow("Scale (multiplier):", scale_spin)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            session.color = color_value
            session.weight = weight_spin.value()
            session.scale = scale_spin.value()
            self.update_file_list_ui()
            self.update_plot()


    def populate_axis_choices(self):
        """Fill the X/Y combos with available columns and pick sensible defaults."""
        if not self.sessions:
            self.x_combo.clear()
            self.y_combo.clear()
            return

        self._suppress_combo = True
        prev_x = self.x_combo.currentText()
        prev_y = self.y_combo.currentText()
        self.x_combo.clear()
        self.y_combo.clear()

        units_lookup = {}
        ordered_columns = []
        for session in self.sessions:
            for name, unit in session.units.items():
                if name not in units_lookup:
                    units_lookup[name] = unit
                    ordered_columns.append(name)
            for name in session.columns.keys():
                if name not in units_lookup:
                    units_lookup[name] = ""
                    ordered_columns.append(name)

        def display_label(name: str) -> str:
            unit = units_lookup.get(name, "")
            return f"{name} ({unit})" if unit else name

        self.display_to_column = {}
        display_names = []
        for col in ordered_columns:
            label = display_label(col)
            display_names.append(label)
            self.display_to_column[label] = col

        for label in display_names:
            self.x_combo.addItem(label)
            self.y_combo.addItem(label)

        default_x = next(
            (lbl for lbl in display_names if self.display_to_column[lbl].lower() == "field"),
            display_names[0],
        )
        default_y = next(
            (
                lbl
                for lbl in display_names
                if self.display_to_column[lbl].lower() in {"x", "y"} and lbl != default_x
            ),
            None,
        )
        if default_y is None:
            default_y = display_names[1] if len(display_names) > 1 else display_names[0]

        self.x_combo.setCurrentText(prev_x if prev_x in self.display_to_column else default_x)
        self.y_combo.setCurrentText(prev_y if prev_y in self.display_to_column else default_y)

        self._last_x_display = self.x_combo.currentText()
        self._last_y_display = self.y_combo.currentText()
        self._suppress_combo = False

    def on_x_changed(self, new_display: str):
        """Handle X combo changes, swapping axes if the selection collides with Y."""
        if self._suppress_combo:
            return
        current_y = self.y_combo.currentText()
        if new_display and new_display == current_y:
            # Keep axes distinct by swapping when the same item is chosen
            self._suppress_combo = True
            self.y_combo.setCurrentText(self._last_x_display)
            self._suppress_combo = False
        self._last_x_display = self.x_combo.currentText()
        self._last_y_display = self.y_combo.currentText()
        self.update_plot()

    def on_y_changed(self, new_display: str):
        """Handle Y combo changes, swapping axes if the selection collides with X."""
        if self._suppress_combo:
            return
        current_x = self.x_combo.currentText()
        if new_display and new_display == current_x:
            # Keep axes distinct by swapping when the same item is chosen
            self._suppress_combo = True
            self.x_combo.setCurrentText(self._last_y_display)
        self._suppress_combo = False
        self._last_x_display = self.x_combo.currentText()
        self._last_y_display = self.y_combo.currentText()
        self.update_plot()

    def on_derivative_toggled(self, state: int):
        """Toggle derivative view; only one secondary series is shown at a time."""
        if state:
            self.integral_checkbox.setChecked(False)
        self.update_plot()

    def on_integral_toggled(self, state: int):
        """Toggle integral view; only one secondary series is shown at a time."""
        if state:
            self.derivative_checkbox.setChecked(False)
        self.update_plot()

    def on_phase_dial_changed(self, value: int):
        """Sync dial changes to spinbox and update plot."""
        if self._suppress_phase:
            return
        self.set_phase(float(value), source="dial")

    def on_phase_spin_changed(self, value: float):
        """Sync spinbox changes to dial and update plot."""
        if self._suppress_phase:
            return
        self.set_phase(float(value), source="spin")

    def on_guess_phase(self):
        """Guess phase angle and apply it."""
        guess = self.suggest_phase_angle()
        self.set_phase(guess)

    def on_reset_phase(self):
        """Reset phase angle to zero."""
        self.set_phase(0.0)

    def on_phase_plus_90(self):
        """Increment phase by +90 degrees."""
        current = self.get_current_phase_deg()
        self.set_phase(current + 90.0)

    def on_phase_minus_90(self):
        """Decrement phase by -90 degrees."""
        current = self.get_current_phase_deg()
        self.set_phase(current - 90.0)

    def on_sweep_error_changed(self, value: float):
        """Update sweep error for the selected session and redraw."""
        session = self.get_selected_session()
        if session is None:
            return
        session.sweep_error = value
        self.update_plot()

    def update_plot(self):
        """Render the current column selections onto the plot widget."""
        if not self.sessions:
            self.plot_widget.clear()
            self._clear_secondary_curves()
            self.plot_widget.getPlotItem().getAxis("right").setLabel("")
            return
        x_label = self.x_combo.currentText()
        y_label = self.y_combo.currentText()
        x_col = self.display_to_column.get(x_label)
        y_col = self.display_to_column.get(y_label)
        if not x_col or not y_col:
            return

        self.plot_widget.clear()
        self._clear_secondary_curves()

        def unit_for(name: str) -> str:
            for session in self.sessions:
                if name in session.units:
                    return session.units.get(name, "")
            return ""

        def axis_label(name: str) -> str:
            unit = unit_for(name)
            return f"{name} ({unit})" if unit else name

        self.plot_widget.setLabel("bottom", axis_label(x_col))
        self.plot_widget.setLabel("left", axis_label(y_col))

        any_secondary = False
        secondary_label = ""
        secondary_unit = ""

        for idx, session in enumerate(self.sessions):
            x_values = self.get_plot_values(session, x_col)
            y_values = self.get_plot_values(session, y_col)
            if not x_values or not y_values:
                continue
            length = min(len(x_values), len(y_values))
            if length == 0:
                continue
            x_values = x_values[:length]
            y_values = y_values[:length]

            scaled_y = [val * session.scale for val in y_values]
            color = session.color or DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
            pen = pg.mkPen(color=color, width=session.weight)
            self.plot_widget.plot(
                x_values,
                scaled_y,
                pen=pen,
                symbol=None,
                symbolSize=4,
            )

            secondary_values = []
            if self.derivative_checkbox.isChecked():
                secondary_values = self.compute_derivative(x_values, scaled_y)
                secondary_label = f"d{y_col}/d{x_col}"
                secondary_unit = self.format_derivative_unit(unit_for(y_col), unit_for(x_col))
            elif self.integral_checkbox.isChecked():
                secondary_values = self.compute_integral(x_values, scaled_y)
                secondary_label = f"Integral of {y_col} d{x_col}"
                secondary_unit = self.format_integral_unit(unit_for(y_col), unit_for(x_col))
            if secondary_values and any(math.isfinite(val) for val in secondary_values):
                secondary_pen = pg.mkPen(color=color, width=max(1, session.weight - 1), style=QtCore.Qt.DashLine)
                item = pg.PlotDataItem(
                    x_values, secondary_values, pen=secondary_pen, name=f"secondary-{idx}"
                )
                self.secondary_vb.addItem(item)
                self._secondary_curves.append(item)
                any_secondary = True

        if any_secondary:
            self.plot_widget.getPlotItem().getAxis("right").setLabel(
                secondary_label, units=secondary_unit
            )
        else:
            self.plot_widget.getPlotItem().getAxis("right").setLabel("")
        self.update_titles()

    def _clear_secondary_curves(self):
        """Remove previous derivative/integral plots, if any."""
        for item in self._secondary_curves:
            try:
                self.secondary_vb.removeItem(item)
            except Exception:
                pass
        self._secondary_curves = []
        self.plot_widget.getPlotItem().getAxis("right").setLabel("")

    @staticmethod
    def compute_derivative(x_values, y_values):
        """Central difference derivative, falling back to forward/backward edges."""
        if len(x_values) < 2:
            return []
        derivative = []
        for idx in range(len(x_values)):
            if idx == 0:
                # Forward difference at the start
                dx = x_values[1] - x_values[0]
                dy = y_values[1] - y_values[0]
            elif idx == len(x_values) - 1:
                # Backward difference at the end
                dx = x_values[-1] - x_values[-2]
                dy = y_values[-1] - y_values[-2]
            else:
                # Central difference in the interior
                dx = x_values[idx + 1] - x_values[idx - 1]
                dy = y_values[idx + 1] - y_values[idx - 1]
            derivative.append(dy / dx if dx != 0 else float("nan"))
        return derivative

    @staticmethod
    def compute_integral(x_values, y_values):
        """Cumulative trapezoidal integral of y with respect to x."""
        if len(x_values) < 2:
            return []
        area = [0.0]
        for idx in range(1, len(x_values)):
            dx = x_values[idx] - x_values[idx - 1]
            # Trapezoid area between successive points
            area.append(area[-1] + 0.5 * (y_values[idx] + y_values[idx - 1]) * dx)
        return area

    @staticmethod
    def compute_direction_and_error(columns):
        """Compute sweep direction (+/-1) and suggested sweep error from Field vs Time."""
        field_key = next((key for key in columns if key.lower() == "field"), None)
        time_key = next((key for key in columns if key.lower() == "time"), None)
        if not field_key or not time_key:
            return [], 0.0
        field_vals = columns[field_key]
        time_vals = columns[time_key]
        if len(field_vals) != len(time_vals) or len(field_vals) < 2:
            return [], 0.0

        raw_derivative = FMRPreview.compute_derivative(time_vals, field_vals)
        smoothed_derivative = FMRPreview._moving_average_static(raw_derivative, window=5)

        direction = []
        for idx, deriv in enumerate(smoothed_derivative):
            if idx < IGNORE_DIRECTION_POINTS:
                direction.append(1)
                continue
            if not math.isfinite(deriv):
                direction.append(1)
            else:
                direction.append(1 if deriv >= 0 else -1)

        sample_idx = min(
            max(int(len(smoothed_derivative) * 0.25), IGNORE_DIRECTION_POINTS),
            len(smoothed_derivative) - 1,
        )
        sample_deriv = smoothed_derivative[sample_idx] if smoothed_derivative else 0.0
        if not math.isfinite(sample_deriv):
            sample_deriv = 0.0
        suggested_error = sample_deriv * SWEEP_ERROR_COEFFICIENT
        return direction, suggested_error

    @staticmethod
    def format_derivative_unit(y_unit: str, x_unit: str) -> str:
        if y_unit and x_unit:
            return f"{y_unit}/{x_unit}"
        if y_unit:
            return f"{y_unit}/x"
        if x_unit:
            return f"1/{x_unit}"
        return ""

    @staticmethod
    def format_integral_unit(y_unit: str, x_unit: str) -> str:
        if y_unit and x_unit:
            return f"{y_unit}*{x_unit}"
        if y_unit:
            return f"{y_unit}*x"
        if x_unit:
            return x_unit
        return ""

    def update_secondary_view_bounds(self):
        """Keep the secondary view box in sync with the primary plot area."""
        plot_item = self.plot_widget.getPlotItem()
        # Mirror geometry so the secondary axis aligns with primary X range
        self.secondary_vb.setGeometry(plot_item.vb.sceneBoundingRect())
        self.secondary_vb.linkedViewChanged(plot_item.vb, self.secondary_vb.XAxis)

    def get_plot_values(self, session: PlotSession, column_name: str):
        """Return column values for a session, applying sweep/phase corrections as needed."""
        values = session.columns.get(column_name, [])
        lower = column_name.lower()

        if lower in {"x", "y"} and session.columns:
            rot_x, rot_y = self.get_rotated_xy(session)
            if rot_x is not None and rot_y is not None:
                values = rot_x if lower == "x" else rot_y
        elif lower == "field" and values:
            if session.direction_data and len(session.direction_data) == len(values):
                shift = session.sweep_error
                values = [val - dir_flag * shift for val, dir_flag in zip(values, session.direction_data)]
        return values

    @staticmethod
    def _moving_average_static(values, window: int):
        """Simple centered moving average smoothing."""
        if window <= 1 or not values:
            return list(values)
        half = window // 2
        smoothed = []
        for idx in range(len(values)):
            start = max(0, idx - half)
            end = min(len(values), idx + half + 1)
            segment = [v for v in values[start:end] if math.isfinite(v)]
            smoothed.append(sum(segment) / len(segment) if segment else 0.0)
        return smoothed

    def get_rotated_xy(self, session: PlotSession):
        """Return rotated X and Y using current phase angle for a session."""
        if session.rotated_xy is not None:
            return session.rotated_xy
        x_key = next((key for key in session.columns if key.lower() == "x"), None)
        y_key = next((key for key in session.columns if key.lower() == "y"), None)
        if not x_key or not y_key:
            session.rotated_xy = (None, None)
            return session.rotated_xy
        x_vals = session.columns.get(x_key, [])
        y_vals = session.columns.get(y_key, [])
        length = min(len(x_vals), len(y_vals))
        if length == 0:
            session.rotated_xy = (None, None)
            return session.rotated_xy
        angle_rad = math.radians(session.phase_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot_x = []
        rot_y = []
        for xv, yv in zip(x_vals[:length], y_vals[:length]):
            if not (math.isfinite(xv) and math.isfinite(yv)):
                rot_x.append(float("nan"))
                rot_y.append(float("nan"))
                continue
            rot_x.append(xv * cos_a - yv * sin_a)
            rot_y.append(xv * sin_a + yv * cos_a)
        session.rotated_xy = (rot_x, rot_y)
        return session.rotated_xy

    def suggest_phase_angle(self):
        """Estimate phase to minimize Y variance for Field >= cutoff."""
        session = self.get_selected_session() or (self.sessions[0] if self.sessions else None)
        x_key = next((key for key in session.columns if key.lower() == "x"), None)
        y_key = next((key for key in session.columns if key.lower() == "y"), None)
        field_key = next((key for key in session.columns if key.lower() == "field"), None)
        if not x_key or not y_key or not field_key:
            return 0.0
        x_vals = session.columns.get(x_key, [])
        y_vals = session.columns.get(y_key, [])
        field_vals = session.columns.get(field_key, [])
        length = min(len(x_vals), len(y_vals), len(field_vals))
        if length == 0:
            return 0.0

        pairs = [
            (x_vals[i], y_vals[i])
            for i in range(length)
            if math.isfinite(field_vals[i]) and field_vals[i] >= PHASE_CUTOFF_FIELD
        ]
        if not pairs:
            pairs = [(x_vals[i], y_vals[i]) for i in range(length)]
        if not pairs:
            return 0.0

        sxx = syy = sxy = 0.0
        for xv, yv in pairs:
            if not (math.isfinite(xv) and math.isfinite(yv)):
                continue
            sxx += xv * xv
            syy += yv * yv
            sxy += xv * yv
        if sxx == 0 and syy == 0:
            return 0.0
        # Choose angle that maximizes X variance (thus minimizing Y variance)
        angle_rad = 0.5 * math.atan2(2 * sxy, sxx - syy)
        return math.degrees(angle_rad)

    def get_selected_session(self):
        if 0 <= self.selected_session_idx < len(self.sessions):
            return self.sessions[self.selected_session_idx]
        return None

    def select_session(self, index: int):
        """Select a session by index and refresh controls/UI."""
        if index < 0 or index >= len(self.sessions):
            self.selected_session_idx = -1
            self.current_file_path = ""
            self.header_display.setPlainText("Header will show after loading a file.")
            self._suppress_phase = True
            self.phase_dial.setValue(0)
            self.phase_spin.setValue(0.0)
            self._suppress_phase = False
            self.sweep_error_spin.blockSignals(True)
            self.sweep_error_spin.setValue(0.0)
            self.sweep_error_spin.blockSignals(False)
            self.update_titles()
            self.update_file_list_ui()
            self.update_plot()
            return

        self.selected_session_idx = index
        session = self.sessions[index]
        self.current_file_path = session.path
        self.header_display.setPlainText(session.header_text or "Header not found.")
        self._suppress_phase = True
        self.phase_dial.setValue(int(round(session.phase_deg)))
        self.phase_spin.setValue(session.phase_deg)
        self._suppress_phase = False
        self.sweep_error_spin.blockSignals(True)
        self.sweep_error_spin.setValue(session.sweep_error)
        self.sweep_error_spin.blockSignals(False)
        self.update_titles()
        self.update_file_list_ui()
        self.update_plot()

    def get_current_phase_deg(self):
        session = self.get_selected_session()
        return session.phase_deg if session else 0.0

    def set_phase(self, angle_deg: float, source: str = None, update: bool = True):
        """Set current phase for the selected session, sync controls, optionally refresh plot."""
        session = self.get_selected_session()
        if session is None:
            return
        clamped = max(-180.0, min(180.0, angle_deg))
        session.phase_deg = clamped
        session.rotated_xy = None
        self._suppress_phase = True
        if source != "dial":
            self.phase_dial.setValue(int(round(clamped)))
        if source != "spin":
            self.phase_spin.setValue(clamped)
        self._suppress_phase = False
        if update:
            self.update_plot()

    def update_titles(self):
        """Update plot title and file display with current phase."""
        count = len(self.sessions)
        selected = self.get_selected_session()
        base_name = os.path.basename(selected.path) if selected else "No file loaded"
        phase_val = self.get_current_phase_deg()
        phase_text = f"(phase {phase_val:.2f}°)"
        if count == 0:
            self.file_path_edit.setText(f"No file loaded {phase_text}")
            self.plot_widget.setTitle("No file loaded")
            self.setWindowTitle("FMR CSV Preview")
            return
        self.file_path_edit.setText(f"{base_name} {phase_text} - {count} file(s) loaded")
        self.plot_widget.setTitle(f"{count} file(s) plotted")
        self.setWindowTitle("FMR CSV Preview")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FMRPreview()
    window.resize(1920, 1200)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
