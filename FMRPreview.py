import csv
import glob
import os
import sys
from PyQt5 import QtWidgets
import pyqtgraph as pg


DATA_DIR_NAME = "Data"


def read_field_vs_x(csv_path: str):
    """Return Field and X columns from the CSV, skipping non-numeric rows."""
    with open(csv_path, newline="") as handle:
        # Strip blank lines so DictReader sees only useful rows
        lines = [line.strip() for line in handle if line.strip()]

    header_index = next((idx for idx, line in enumerate(lines) if line.lower().startswith("time,field")), None)
    if header_index is None:
        raise ValueError("Could not find a 'Time,Field,...' header row in the file.")

    # csv.DictReader maps each remaining row to a dict keyed by the header row
    reader = csv.DictReader(lines[header_index:])
    fields, x_values = [], []
    for row in reader:
        normalized = {key.lower(): value for key, value in row.items() if key}
        try:
            field_val = float(normalized.get("field", ""))
            x_val = float(normalized.get("x", ""))
        except (TypeError, ValueError):
            continue
        fields.append(field_val)
        x_values.append(x_val)

    return fields, x_values


class FMRPreview(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FMR CSV Preview")
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR_NAME)

        # PyQtGraph PlotWidget gives us a fast, interactive plotting canvas
        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)  # draw a light grid
        self.plot_widget.setLabel("left", "X (V)")  # y-axis label
        self.plot_widget.setLabel("bottom", "Field (Oe)")  # x-axis label

        self.file_label = QtWidgets.QLabel("No file loaded")
        # QPushButton handles the click, and we connect its signal to our slot
        self.open_button = QtWidgets.QPushButton("Open CSV...")
        self.open_button.clicked.connect(self.choose_file)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.file_label)
        top_row.addStretch()
        top_row.addWidget(self.open_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.plot_widget)

        container = QtWidgets.QWidget()
        container.setLayout(layout)  # QWidget owns the layout and children
        self.setCentralWidget(container)  # QMainWindow API to show our UI tree

        self.load_first_csv_if_present()

    def choose_file(self):
        start_dir = self.data_dir if os.path.isdir(self.data_dir) else os.getcwd()
        # QFileDialog prompts the user to pick a CSV from disk
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
            fields, x_values = read_field_vs_x(path)
        except Exception as exc:  # noqa: BLE001
            # QMessageBox shows a modal warning dialog for load failures
            QtWidgets.QMessageBox.warning(self, "Load error", str(exc))
            return

        if not fields:
            # QMessageBox information dialog for empty datasets
            QtWidgets.QMessageBox.information(self, "No data", "No numeric Field/X rows found.")
            return

        self.plot_widget.clear()  # remove any prior curves
        pen = pg.mkPen(color=(200, 50, 50), width=2)  # styled line pen
        self.plot_widget.plot(fields, x_values, pen=pen, symbol="o", symbolSize=4, symbolBrush=(60, 60, 200))
        self.plot_widget.setTitle(os.path.basename(path))  # show filename above the plot
        self.file_label.setText(path)


def main():
    app = QtWidgets.QApplication(sys.argv)  # Qt application event loop holder
    window = FMRPreview()
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec_())  # hand control to Qt's event loop


if __name__ == "__main__":
    main()

