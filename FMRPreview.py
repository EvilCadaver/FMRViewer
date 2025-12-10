import csv
import glob
import os
import sys
from PyQt5 import QtWidgets
import pyqtgraph as pg


DATA_DIR_NAME = "Data"


def parse_numeric_columns(csv_path: str):
    """
    Parse a CSV where the header row lists column names and the next row lists units.
    Returns dicts: columns[name] -> list[float], units[name] -> unit string (or "").
    """
    with open(csv_path, newline="") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    header_index = next(
        (idx for idx, line in enumerate(lines) if "," in line and "field" in line.lower()),
        None,
    )
    if header_index is None:
        header_index = next((idx for idx, line in enumerate(lines) if "," in line), None)
    if header_index is None:
        raise ValueError("Could not find a header row with comma-separated columns.")

    reader = csv.reader(lines[header_index:])
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("File ended before reading header.")  # noqa: TRY200

    units_row = next(reader, None)
    if units_row and len(units_row) != len(header):
        units_row = None  # ignore a mismatched units row

    data_rows = list(reader)

    columns = {name.strip(): [] for name in header}
    units = {
        name.strip(): (units_row[idx].strip() if units_row and idx < len(units_row) else "")
        for idx, name in enumerate(header)
    }

    for row in data_rows:
        if len(row) < len(header):
            continue
        try:
            values = [float(value) for value in row[: len(header)]]
        except ValueError:
            continue
        for name, value in zip(header, values):
            columns[name.strip()].append(value)

    columns = {name: vals for name, vals in columns.items() if vals}
    units = {name: units.get(name, "") for name in columns}
    return columns, units


class FMRPreview(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FMR CSV Preview")
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR_NAME)
        self.columns = {}
        self.units = {}
        self.display_to_column = {}
        self._suppress_combo = False
        self._last_x_display = ""
        self._last_y_display = ""

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "X (V)")
        self.plot_widget.setLabel("bottom", "Field (Oe)")

        self.file_label = QtWidgets.QLabel("No file loaded")
        self.open_button = QtWidgets.QPushButton("Open CSV...")
        self.open_button.clicked.connect(self.choose_file)

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
        pattern = os.path.join(self.data_dir, "*.csv")
        csv_files = sorted(glob.glob(pattern))
        if csv_files:
            self.load_and_plot(csv_files[0])

    def load_and_plot(self, path: str):
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
        self.populate_axis_choices()
        self.update_plot()

        self.plot_widget.setTitle(os.path.basename(path))
        self.file_label.setText(path)

    def populate_axis_choices(self):
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

        self.x_combo.setCurrentText(default_x)
        self.y_combo.setCurrentText(default_y)

        self._last_x_display = self.x_combo.currentText()
        self._last_y_display = self.y_combo.currentText()
        self._suppress_combo = False

    def on_x_changed(self, new_display: str):
        if self._suppress_combo:
            return
        current_y = self.y_combo.currentText()
        if new_display and new_display == current_y:
            self._suppress_combo = True
            self.y_combo.setCurrentText(self._last_x_display)
            self._suppress_combo = False
        self._last_x_display = self.x_combo.currentText()
        self._last_y_display = self.y_combo.currentText()
        self.update_plot()

    def on_y_changed(self, new_display: str):
        if self._suppress_combo:
            return
        current_x = self.x_combo.currentText()
        if new_display and new_display == current_x:
            self._suppress_combo = True
            self.x_combo.setCurrentText(self._last_y_display)
            self._suppress_combo = False
        self._last_x_display = self.x_combo.currentText()
        self._last_y_display = self.y_combo.currentText()
        self.update_plot()

    def update_plot(self):
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
