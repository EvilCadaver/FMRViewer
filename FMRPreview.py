import csv
import glob
import math
import os
import sys
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# Lightweight CSV plotting helper with optional derivative/integral overlay

DATA_DIR_NAME = "Data"


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

        # File label + button to pick a CSV
        self.file_label = QtWidgets.QLabel("No file loaded")
        self.open_button = QtWidgets.QPushButton("Open CSV...")
        self.open_button.clicked.connect(self.choose_file)

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

        options_row = QtWidgets.QHBoxLayout()
        options_row.addWidget(self.derivative_checkbox)
        options_row.addSpacing(12)
        options_row.addWidget(self.integral_checkbox)
        options_row.addStretch()

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.file_label)
        top_row.addStretch()
        top_row.addWidget(self.open_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(axes_row)
        layout.addLayout(options_row)
        layout.addWidget(self.plot_widget)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_first_csv_if_present()

    def choose_file(self):
        """Open a file dialog and load the chosen CSV."""
        start_dir = self.data_dir if os.path.isdir(self.data_dir) else os.getcwd()
        # Let the user pick a CSV; remember start directory preference
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            start_dir,
            "CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.load_and_plot(path)

    def load_first_csv_if_present(self):
        """Auto-load the first CSV in Data/ so the window is immediately useful."""
        pattern = os.path.join(self.data_dir, "*.csv")
        csv_files = sorted(glob.glob(pattern))
        if csv_files:
            self.load_and_plot(csv_files[0])

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
        self.populate_axis_choices()  # fills combos and sets defaults
        self.update_plot()  # draw initial plot based on defaults

        self.plot_widget.setTitle(os.path.basename(path))
        self.file_label.setText(path)

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

    def update_plot(self):
        """Render the current column selections onto the plot widget."""
        if not self.columns:
            return
        x_label = self.x_combo.currentText()
        y_label = self.y_combo.currentText()
        x_col = self.display_to_column.get(x_label)
        y_col = self.display_to_column.get(y_label)
        if not x_col or not y_col:
            return

        x_values = self.columns.get(x_col, [])
        y_values = self.columns.get(y_col, [])
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
            symbol="o",
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FMRPreview()
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
