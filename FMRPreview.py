import csv
import glob
import os
import sys
from PyQt5 import QtWidgets
import pyqtgraph as pg


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
        # Remove blank lines so the CSV reader only sees meaningful rows
        lines = [line.strip() for line in handle if line.strip()]

    # Pick a header row. Prefer a row containing "field" to match the provided data,
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

    # Parse numeric rows; any row that fails float conversion is skipped
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
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR_NAME)

        # In-memory state for the loaded CSV
        self.columns = {}  # column name -> list[float]
        self.units = {}  # column name -> unit string
        self.display_to_column = {}  # "Field (Oe)" -> "Field"
        self._suppress_combo = False  # guard to prevent signal loops while we adjust combos
        self._last_x_display = ""  # previous X selection (for swap logic)
        self._last_y_display = ""  # previous Y selection (for swap logic)

        # PyQtGraph plot surface
        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "X (V)")
        self.plot_widget.setLabel("bottom", "Field (Oe)")

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

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.file_label)
        top_row.addStretch()
        top_row.addWidget(self.open_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(axes_row)
        layout.addWidget(self.plot_widget)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_first_csv_if_present()

    def choose_file(self):
        """Open a file dialog and load the chosen CSV."""
        start_dir = self.data_dir if os.path.isdir(self.data_dir) else os.getcwd()
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

        self.plot_widget.clear()
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FMRPreview()
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
