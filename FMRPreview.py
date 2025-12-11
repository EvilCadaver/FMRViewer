import csv
import glob
import math
import os
import sys
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# Lightweight CSV plotting helper with optional derivative/integral overlay

DATA_DIR_NAME = "Data"
SWEEP_ERROR_COEFFICIENT = 0.38
IGNORE_DIRECTION_POINTS = 10  # points at start assumed positive sweep
PHASE_CUTOFF_FIELD = 500  # Oe to estimate initial phase


def parse_numeric_columns(csv_path: str):
    """
    Parse a CSV file that uses the pattern:
        header row: column names (e.g., Time,Field,X,Y,R,theta)
        units row : units for each column (e.g., s,Oe,V,V,V,deg)
        data rows : numeric values

    Returns two dicts:
        columns[name] -> list of float values
        units[name]   -> unit string (or empty if missing)
    """
    with open(csv_path, newline="") as handle:
        # Strip out empty lines so the CSV reader only sees meaningful rows
        lines = [line.strip() for line in handle if line.strip()]

    # Pick a header row. Prefer a row containing "field" to match typical data,
    # otherwise fall back to the first comma-separated row.
    header_index = next(
        (idx for idx, line in enumerate(lines) if "," in line and "field" in line.lower()),
        None,
    )
    if header_index is None:
        header_index = next((idx for idx, line in enumerate(lines) if "," in line), None)
    if header_index is None:
        raise ValueError("Could not find a header row with comma-separated columns.")

    # Create a CSV reader from the header forward; rows are simple lists at this point.
    reader = csv.reader(lines[header_index:])
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
    return columns, units


class FMRPreview(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FMR CSV Preview")
        # Default data directory lives alongside the script in Data/
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR_NAME)

        # In-memory state for the loaded CSV
        self.columns = {}  # column name -> list[float]
        self.units = {}  # column name -> unit string
        self.display_to_column = {}  # "Field (Oe)" -> "Field"
        self._suppress_combo = False  # guard to prevent signal loops while we adjust combos
        self._last_x_display = ""  # previous X selection (for swap logic)
        self._last_y_display = ""  # previous Y selection (for swap logic)
        self._secondary_curve = None  # plot item for derivative/integral overlay
        self.direction_data = []  # derived +1/-1 sweep direction
        self.current_phase_deg = 0.0
        self._suppress_phase = False
        self.current_file_path = ""
        self._rotated_xy = None

        # PyQtGraph plot surface
        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "X (V)")
        self.plot_widget.setLabel("bottom", "Field (Oe)")
        # Attach a secondary ViewBox for the right-side axis
        plot_item = self.plot_widget.getPlotItem()
        plot_item.showAxis("right")
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
        phase_spin_row.addWidget(QtWidgets.QLabel("Phase shift (deg):"))
        phase_spin_row.addWidget(self.phase_spin)
        phase_spin_row.addWidget(self.guess_phase_button)
        phase_spin_row.addWidget(self.reset_phase_button)
        # phase_spin_row.addStretch()

        dial_buttons_col = QtWidgets.QVBoxLayout()
        # dial_buttons_col.addStretch()
        dial_buttons_col.addWidget(self.phase_plus_button)
        dial_buttons_col.addSpacing(6)
        dial_buttons_col.addWidget(self.phase_minus_button)
        # dial_buttons_col.addStretch()

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
        self.sweep_error_spin.valueChanged.connect(self.update_plot)

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

        left_col = QtWidgets.QVBoxLayout()
        left_col.addLayout(axes_row)
        left_col.addLayout(options_row)
        left_col.addLayout(sweep_row)
        left_col.addLayout(phase_spin_row)
        left_col.addLayout(phase_dial_row)
        left_col.addStretch()

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.addLayout(left_col)
        controls_row.addSpacing(12)
        controls_row.addStretch()

        body_row = QtWidgets.QHBoxLayout()
        body_row.addWidget(self.plot_widget, 1)
        body_row.addLayout(controls_row)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(body_row)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_first_csv_if_present()

    def choose_file(self):
        """Open a file dialog and load the chosen CSV."""
        start_dir = (
            self.last_viewed_dir
            or self.find_latest_data_folder()
            or (self.data_dir if os.path.isdir(self.data_dir) else os.getcwd())
        )
        # Let the user pick a CSV; remember start directory preference
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            start_dir,
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.load_and_plot(path)

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
            self.load_and_plot(newest_csv)

    def load_and_plot(self, path: str):
        """Parse a file, populate selectors, and render the plot."""
        try:
            columns, units = parse_numeric_columns(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Load error", str(exc))
            return

        if not columns:
            QtWidgets.QMessageBox.information(self, "No data", "No numeric columns found.")
            return

        # Commit parsed data, populate UI, and render
        self.columns = columns
        self.units = units
        self.direction_data = []
        self.prepare_sweep_direction_and_error()
        self.current_file_path = path
        self.set_phase(0.0, update=False)
        self.populate_axis_choices()  # fills combos and sets defaults
        self.update_plot()  # draw initial plot based on defaults

        self.last_viewed_dir = os.path.dirname(path)

    def populate_axis_choices(self):
        """Fill the X/Y combos with available columns and pick sensible defaults."""
        self._suppress_combo = True
        self.x_combo.clear()
        self.y_combo.clear()

        def display_label(name: str) -> str:
            # Compose "Name (unit)" when a unit exists
            unit = self.units.get(name, "")
            return f"{name} ({unit})" if unit else name

        self.display_to_column = {}
        display_names = []
        for col in self.columns:
            label = display_label(col)
            display_names.append(label)
            self.display_to_column[label] = col

        for label in display_names:
            self.x_combo.addItem(label)
            self.y_combo.addItem(label)

        # Default X: prefer "Field" if present; otherwise first column
        default_x = next(
            (lbl for lbl in display_names if self.display_to_column[lbl].lower() == "field"),
            display_names[0],
        )
        # Default Y: prefer "X" or "Y" (whichever is available and not the same as X)
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

        self.x_combo.setCurrentText(default_x)
        self.y_combo.setCurrentText(default_y)

        self._last_x_display = self.x_combo.currentText()
        self._last_y_display = self.y_combo.currentText()
        self._suppress_combo = False
        self.derivative_checkbox.setChecked(False)
        self.integral_checkbox.setChecked(False)

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
        self.set_phase(self.current_phase_deg + 90.0)

    def on_phase_minus_90(self):
        """Decrement phase by -90 degrees."""
        self.set_phase(self.current_phase_deg - 90.0)

    def update_plot(self):
        """Render the current column selections onto the plot widget."""
        if not self.columns:
            return
        self._rotated_xy = None  # clear rotation cache for this draw
        x_label = self.x_combo.currentText()
        y_label = self.y_combo.currentText()
        x_col = self.display_to_column.get(x_label)
        y_col = self.display_to_column.get(y_label)
        if not x_col or not y_col:
            return

        x_values = self.get_plot_values(x_col)
        y_values = self.get_plot_values(y_col)
        if not x_values or not y_values:
            return

        # Draw primary series on the left axis
        self.plot_widget.clear()
        self._clear_secondary_curve()
        pen = pg.mkPen(color=(200, 50, 50), width=2)
        self.plot_widget.plot(
            x_values,
            y_values,
            pen=pen,
            symbol=None,
            symbolSize=4,
            symbolBrush=(60, 60, 200),
        )

        def axis_label(name: str) -> str:
            unit = self.units.get(name, "")
            return f"{name} ({unit})" if unit else name

        self.plot_widget.setLabel("bottom", axis_label(x_col))
        self.plot_widget.setLabel("left", axis_label(y_col))

        # Build secondary series (derivative or integral) if requested
        secondary_values = []
        secondary_label = ""
        secondary_unit = ""
        if self.derivative_checkbox.isChecked():
            secondary_values = self.compute_derivative(x_values, y_values)
            secondary_label = f"d{y_col}/d{x_col}"
            secondary_unit = self.format_derivative_unit(
                self.units.get(y_col, ""), self.units.get(x_col, "")
            )
        elif self.integral_checkbox.isChecked():
            secondary_values = self.compute_integral(x_values, y_values)
            secondary_label = f"Integral of {y_col} d{x_col}"
            secondary_unit = self.format_integral_unit(
                self.units.get(y_col, ""), self.units.get(x_col, "")
            )

        if secondary_values and any(math.isfinite(val) for val in secondary_values):
            secondary_pen = pg.mkPen(color=(50, 100, 200), width=2, style=QtCore.Qt.DashLine)
            self._secondary_curve = pg.PlotDataItem(
                x_values, secondary_values, pen=secondary_pen, name="secondary"
            )
            self.secondary_vb.addItem(self._secondary_curve)
            self.plot_widget.getPlotItem().getAxis("right").setLabel(
                secondary_label, units=secondary_unit
            )
        else:
            self.plot_widget.getPlotItem().getAxis("right").setLabel("")

        self.update_titles()

    def _clear_secondary_curve(self):
        """Remove the previous derivative/integral plot, if any."""
        if self._secondary_curve is not None:
            self.secondary_vb.removeItem(self._secondary_curve)
            self._secondary_curve = None
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

    def get_plot_values(self, column_name: str):
        """Return column values, applying sweep/phase corrections as needed."""
        values = self.columns.get(column_name, [])
        lower = column_name.lower()

        if lower in {"x", "y"} and self.columns:
            rot_x, rot_y = self.get_rotated_xy()
            if rot_x is not None and rot_y is not None:
                values = rot_x if lower == "x" else rot_y
        elif lower == "field" and values:
            if self.direction_data and len(self.direction_data) == len(values):
                shift = self.sweep_error_spin.value()
                values = [val - dir_flag * shift for val, dir_flag in zip(values, self.direction_data)]
        return values

    def prepare_sweep_direction_and_error(self):
        """Compute sweep direction (+/-1) and suggested sweep error from Field vs Time."""
        field_key = next((key for key in self.columns if key.lower() == "field"), None)
        time_key = next((key for key in self.columns if key.lower() == "time"), None)
        if not field_key or not time_key:
            self.direction_data = []
            self.sweep_error_spin.setValue(0.0)
            return
        field_vals = self.columns[field_key]
        time_vals = self.columns[time_key]
        if len(field_vals) != len(time_vals) or len(field_vals) < 2:
            self.direction_data = []
            self.sweep_error_spin.setValue(0.0)
            return

        raw_derivative = self.compute_derivative(time_vals, field_vals)
        smoothed_derivative = self._moving_average(raw_derivative, window=5)

        # Build direction flags, forcing early points to +1
        direction = []
        for idx, deriv in enumerate(smoothed_derivative):
            if idx < IGNORE_DIRECTION_POINTS:
                direction.append(1)
                continue
            if not math.isfinite(deriv):
                direction.append(1)
            else:
                direction.append(1 if deriv >= 0 else -1)

        # Suggest sweep error from stable region near 25% of data
        sample_idx = min(max(int(len(smoothed_derivative) * 0.25), IGNORE_DIRECTION_POINTS), len(smoothed_derivative) - 1)
        sample_deriv = smoothed_derivative[sample_idx] if smoothed_derivative else 0.0
        if not math.isfinite(sample_deriv):
            sample_deriv = 0.0
        suggested_error = sample_deriv * SWEEP_ERROR_COEFFICIENT

        self.direction_data = direction
        self.columns["Direction"] = direction
        self.units["Direction"] = ""
        self.sweep_error_spin.blockSignals(True)
        self.sweep_error_spin.setValue(suggested_error)
        self.sweep_error_spin.blockSignals(False)

    @staticmethod
    def _moving_average(values, window: int):
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

    def get_rotated_xy(self):
        """Return rotated X and Y using current phase angle."""
        if self._rotated_xy is not None:
            return self._rotated_xy
        x_key = next((key for key in self.columns if key.lower() == "x"), None)
        y_key = next((key for key in self.columns if key.lower() == "y"), None)
        if not x_key or not y_key:
            self._rotated_xy = (None, None)
            return self._rotated_xy
        x_vals = self.columns.get(x_key, [])
        y_vals = self.columns.get(y_key, [])
        length = min(len(x_vals), len(y_vals))
        if length == 0:
            self._rotated_xy = (None, None)
            return self._rotated_xy
        angle_rad = math.radians(self.current_phase_deg)
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
        self._rotated_xy = (rot_x, rot_y)
        return self._rotated_xy

    def suggest_phase_angle(self):
        """Estimate phase to minimize Y variance for Field >= cutoff."""
        x_key = next((key for key in self.columns if key.lower() == "x"), None)
        y_key = next((key for key in self.columns if key.lower() == "y"), None)
        field_key = next((key for key in self.columns if key.lower() == "field"), None)
        if not x_key or not y_key or not field_key:
            return 0.0
        x_vals = self.columns.get(x_key, [])
        y_vals = self.columns.get(y_key, [])
        field_vals = self.columns.get(field_key, [])
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

    def set_phase(self, angle_deg: float, source: str = None, update: bool = True):
        """Set current phase, keep controls in sync, optionally refresh plot."""
        clamped = max(min(angle_deg, 180.0), -180.0)
        self.current_phase_deg = clamped
        self._rotated_xy = None
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
        base = os.path.basename(self.current_file_path) if self.current_file_path else "No file loaded"
        phase_text = f"(phase {self.current_phase_deg:.2f}°)"
        display_path = self.current_file_path or "No file loaded"
        self.file_path_edit.setText(f"{display_path} {phase_text}")
        self.plot_widget.setTitle(base)
        self.setWindowTitle("FMR CSV Preview")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FMRPreview()
    window.resize(1920, 1200)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
