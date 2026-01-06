from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

ArrayLike = np.ndarray | list[float] | tuple[float, ...]
ComplexArray = np.ndarray


def _as_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _get_param(params: Any, key: str, default: float | None = None) -> float:
    if isinstance(params, Mapping):
        value = params.get(key, default)
    else:
        value = getattr(params, key, default)
    if value is None:
        raise KeyError(f"Missing parameter '{key}'.")
    return float(value)


def _mu_eff_surface_impedance(field_koe: ArrayLike, frequency_ghz: float, params: Any) -> ComplexArray:
    field = _as_float_array(field_koe)
    h_eff = field + _get_param(params, "Hk", 0.0)
    b_eff = h_eff + _get_param(params, "Js")

    alpha = _get_param(params, "alpha")
    gamma = _get_param(params, "gamma")
    g_factor = _get_param(params, "g")
    omg = float(frequency_ghz) / (gamma * g_factor)

    numerator = (omg**2) - (b_eff + 1j * alpha * omg) ** 2
    denominator = (omg**2) - (h_eff + 1j * alpha * omg) * (b_eff + 1j * alpha * omg)
    return numerator / denominator


def _broaden_mu_eff_lognormal(field_koe: ArrayLike, frequency_ghz: float, params: Any) -> ComplexArray:
    """
    Log-normal ("s-space Gaussian") broadening:

        mu_eff_broad(H) = ∫ mu_eff(exp(s) * H) * N(s; 0, sigma_s^2) ds

    computed via Gauss-Hermite quadrature on s = sqrt(2) * sigma_s * t.
    """
    sigma_s = _get_param(params, "sigma_s", 0.0)
    if sigma_s <= 0.0:
        return _mu_eff_surface_impedance(field_koe, frequency_ghz, params)

    n = int(round(_get_param(params, "broadening_n", 31.0)))
    n = max(n, 3)
    t, w = np.polynomial.hermite.hermgauss(n)  # ∫ exp(-t^2) f(t) dt ≈ Σ w_i f(t_i)
    scale = np.exp(np.sqrt(2.0) * sigma_s * t)  # exp(s)

    field = _as_float_array(field_koe)
    scale_nd = scale.reshape((scale.size,) + (1,) * field.ndim)
    scaled_fields = scale_nd * field
    mu = _mu_eff_surface_impedance(scaled_fields, frequency_ghz, params)
    w_nd = w.reshape((w.size,) + (1,) * field.ndim)
    return (w_nd * mu).sum(axis=0) / np.sqrt(np.pi)


def surface_impedance_response(field_koe: ArrayLike, frequency_ghz: float, params: Any) -> ComplexArray:
    """
    Surface impedance (thick limit) for a good conductor:

        Zs = (1 + i) * sqrt( ω μ0 μ_eff ρ / 2 )

    where μ_eff is treated as a relative (dimensionless) complex effective permeability.
    """
    mu_eff = _broaden_mu_eff_lognormal(field_koe, frequency_ghz, params)

    mu0 = 4.0e-7 * np.pi  # H/m
    rho_ohm_m = _get_param(params, "rho") * 1e-8  # uOhm*cm -> ohm*m
    omega = 2.0 * np.pi * (float(frequency_ghz) * 1e9)
    return (1.0 + 1j) * np.sqrt(omega * mu0 * mu_eff * rho_ohm_m / 2.0)


def surface_impedance_absorbed_power(field_koe: ArrayLike, frequency_ghz: float, params: Any) -> np.ndarray:
    field = _as_float_array(field_koe)
    zs = surface_impedance_response(field, frequency_ghz, params)
    field_power = _get_param(params, "field_power", 2.0)
    return np.real(zs) * (field**field_power)


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    default: float
    unit: str


DEFAULT_PARAMETER_SPECS: Sequence[ParameterSpec] = (
    ParameterSpec("Js", "Saturation polarization", 6, "kG"),
    ParameterSpec("g", "Gyromagnetic factor", 2, ""),
    ParameterSpec("alpha", "Damping", 1.35e-3, ""),
    ParameterSpec("rho", "Resistivity", 9.7, "uOhm*cm"),
    ParameterSpec("Hk", "Anisotropy field", 0.0, "kOe"),
    ParameterSpec("sigma_s", "Broadening sigma_s", 0.0, ""),
    ParameterSpec("broadening_n", "Broadening points", 31.0, ""),
    ParameterSpec("f", "Frequency", 36.0, "GHz"),
    ParameterSpec("gamma", "Gamma", 1.399611, "GHz/kOe"),
    ParameterSpec("field_power", "Field power", 2.0, ""),
)


def launch_ui() -> None:
    import os

    from PyQt5 import QtCore, QtWidgets
    import pyqtgraph as pg

    class FMRFittingWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("FMR Surface Impedance Fitting")

            self.data_path: str = ""
            self.field_values_koe: Optional[np.ndarray] = None
            self.y_values: Optional[np.ndarray] = None
            self.y_label: str = ""
            self._suggested_range: Optional[tuple[float, float, int]] = None

            self.plot_widget = pg.PlotWidget(background="w")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLabel("bottom", "Field (kOe)")
            self.plot_widget.setLabel("left", "Signal")
            self.plot_item = self.plot_widget.getPlotItem()

            self.data_pen = pg.mkPen(color=(60, 130, 220), width=1)
            self.model_pen = pg.mkPen(color=(220, 70, 70), width=2)

            self.experimental_curve = pg.PlotDataItem(
                pen=None,
                symbol="o",
                symbolSize=5,
                symbolPen=self.data_pen,
                symbolBrush=(60, 130, 220, 60),
            )
            self.plot_item.addItem(self.experimental_curve)

            self.model_curve = pg.PlotDataItem(pen=self.model_pen)
            self.plot_item.addItem(self.model_curve)

            self.legend = self.plot_item.addLegend()
            self.legend.addItem(self.experimental_curve, "Experimental")
            self.legend.addItem(self.model_curve, "Model")

            self.file_path_edit = QtWidgets.QLineEdit("No file loaded")
            self.file_path_edit.setReadOnly(True)
            self.open_button = QtWidgets.QPushButton("Open CSV...")
            self.open_button.clicked.connect(self.choose_file)

            self.field_combo = QtWidgets.QComboBox()
            self.y_combo = QtWidgets.QComboBox()
            self.field_combo.currentIndexChanged.connect(self._on_column_changed)
            self.y_combo.currentIndexChanged.connect(self._on_column_changed)

            self.model_field_start = QtWidgets.QDoubleSpinBox()
            self.model_field_start.setRange(-1e6, 1e6)
            self.model_field_start.setDecimals(6)
            self.model_field_start.setValue(0.5)
            self.model_field_start.valueChanged.connect(self.update_plot)

            self.model_field_stop = QtWidgets.QDoubleSpinBox()
            self.model_field_stop.setRange(-1e6, 1e6)
            self.model_field_stop.setDecimals(6)
            self.model_field_stop.setValue(20.0)
            self.model_field_stop.valueChanged.connect(self.update_plot)

            self.model_field_points = QtWidgets.QSpinBox()
            self.model_field_points.setRange(2, 1_000_000)
            self.model_field_points.setValue(2001)
            self.model_field_points.valueChanged.connect(self.update_plot)

            self.range_suggestion_label = QtWidgets.QLabel("Suggested model range: (load data)")
            self.range_suggestion_label.setWordWrap(True)
            self.apply_suggested_range_button = QtWidgets.QPushButton("Use suggested range")
            self.apply_suggested_range_button.setEnabled(False)
            self.apply_suggested_range_button.clicked.connect(self._apply_suggested_range)

            self.model_offset = QtWidgets.QDoubleSpinBox()
            self.model_offset.setDecimals(8)
            self.model_offset.setRange(-1e12, 1e12)
            self.model_offset.setValue(0.0)
            self.model_offset.valueChanged.connect(self.update_plot)

            self.model_scale = QtWidgets.QDoubleSpinBox()
            self.model_scale.setDecimals(8)
            self.model_scale.setRange(-1e12, 1e12)
            self.model_scale.setValue(1.0)
            self.model_scale.valueChanged.connect(self.update_plot)

            self.derivative_checkbox = QtWidgets.QCheckBox("Use model derivative dP/dH")
            self.derivative_checkbox.setChecked(True)
            self.derivative_checkbox.stateChanged.connect(self.update_plot)

            self.param_inputs: Dict[str, QtWidgets.QDoubleSpinBox] = {}
            self.param_group = self._build_parameter_controls()

            self.update_button = QtWidgets.QPushButton("Update plot")
            self.update_button.clicked.connect(self.update_plot)

            top_row = QtWidgets.QHBoxLayout()
            top_row.addWidget(self.file_path_edit)
            top_row.addWidget(self.open_button)

            col_row = QtWidgets.QHBoxLayout()
            col_row.addWidget(QtWidgets.QLabel("Field:"))
            col_row.addWidget(self.field_combo)
            col_row.addSpacing(8)
            col_row.addWidget(QtWidgets.QLabel("Y:"))
            col_row.addWidget(self.y_combo)
            col_row.addStretch()

            model_range_group = QtWidgets.QGroupBox("Model Field Range (kOe)")
            model_range_layout = QtWidgets.QFormLayout()
            model_range_layout.setLabelAlignment(QtCore.Qt.AlignRight)
            model_range_layout.addRow("Start", self.model_field_start)
            model_range_layout.addRow("Stop", self.model_field_stop)
            model_range_layout.addRow("Points", self.model_field_points)
            model_range_layout.addRow(self.range_suggestion_label)
            model_range_layout.addRow(self.apply_suggested_range_button)
            model_range_group.setLayout(model_range_layout)

            modifier_group = QtWidgets.QGroupBox("Model Linear Modifier: c + k·f(H)")
            modifier_layout = QtWidgets.QFormLayout()
            modifier_layout.setLabelAlignment(QtCore.Qt.AlignRight)
            modifier_layout.addRow("c (offset)", self.model_offset)
            modifier_layout.addRow("k (scale)", self.model_scale)
            modifier_group.setLayout(modifier_layout)

            controls_layout = QtWidgets.QVBoxLayout()
            controls_layout.addLayout(top_row)
            controls_layout.addLayout(col_row)
            controls_layout.addWidget(model_range_group)
            controls_layout.addWidget(modifier_group)
            controls_layout.addWidget(self.derivative_checkbox)
            controls_layout.addWidget(self.param_group)
            controls_layout.addWidget(self.update_button)
            controls_layout.addStretch()

            controls = QtWidgets.QWidget()
            controls.setLayout(controls_layout)
            controls.setMaximumWidth(420)

            main_layout = QtWidgets.QHBoxLayout()
            main_layout.addWidget(controls)
            main_layout.addWidget(self.plot_widget, 1)

            container = QtWidgets.QWidget()
            container.setLayout(main_layout)
            self.setCentralWidget(container)

        def _apply_suggested_range(self) -> None:
            if self._suggested_range is None:
                return
            start, stop, points = self._suggested_range
            self.model_field_start.setValue(start)
            self.model_field_stop.setValue(stop)
            self.model_field_points.setValue(points)

        def _update_range_suggestion(self) -> None:
            if self.field_values_koe is None or self.field_values_koe.size == 0:
                self._suggested_range = None
                self.range_suggestion_label.setText("Suggested model range: (load data)")
                self.apply_suggested_range_button.setEnabled(False)
                return

            finite = self.field_values_koe[np.isfinite(self.field_values_koe)]
            if finite.size == 0:
                self._suggested_range = None
                self.range_suggestion_label.setText("Suggested model range: (no finite field values)")
                self.apply_suggested_range_button.setEnabled(False)
                return

            start = float(np.min(finite))
            stop = float(np.max(finite))
            if start == stop:
                stop = start + 1.0

            suggested_points = int(self.field_values_koe.size)
            suggested_points = max(2, min(1_000_000, suggested_points))

            self._suggested_range = (start, stop, suggested_points)
            self.range_suggestion_label.setText(
                f"Suggested model range: start={start:.6g} kOe, stop={stop:.6g} kOe, points={suggested_points}"
            )
            self.apply_suggested_range_button.setEnabled(True)

        def _build_parameter_controls(self) -> QtWidgets.QGroupBox:
            group = QtWidgets.QGroupBox("Parameters")
            layout = QtWidgets.QFormLayout()
            layout.setLabelAlignment(QtCore.Qt.AlignRight)

            for spec in DEFAULT_PARAMETER_SPECS:
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(8 if abs(spec.default) < 0.01 else 4)
                spin.setRange(-1e12, 1e12)
                spin.setValue(spec.default)
                spin.valueChanged.connect(self.update_plot)
                self.param_inputs[spec.key] = spin
                label = f"{spec.label} [{spec.unit}]" if spec.unit else spec.label
                layout.addRow(label, spin)

            group.setLayout(layout)
            return group

        def _params_dict(self) -> Dict[str, float]:
            return {key: spin.value() for key, spin in self.param_inputs.items()}

        def choose_file(self) -> None:
            start_dir = ".\\Data" if not self.data_path else os.path.dirname(self.data_path)
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open FMR CSV",
                start_dir,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not path:
                return
            self._load_csv(path)

        def _load_csv(self, csv_path: str) -> None:
            try:
                from FMRPreview import parse_numeric_columns
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(
                    self, "Import error", f"Cannot import `parse_numeric_columns` from `FMRPreview.py`:\n{exc}"
                )
                return

            try:
                columns, units, _header = parse_numeric_columns(csv_path)
            except Exception as exc:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to parse CSV:\n{exc}")
                return

            self.data_path = csv_path
            self.file_path_edit.setText(csv_path)

            keys = list(columns.keys())
            if not keys:
                QtWidgets.QMessageBox.warning(self, "No data", "No numeric columns found.")
                return

            def display_name(key: str) -> str:
                unit = str(units.get(key, "") or "").strip()
                return f"{key} ({unit})" if unit else key

            self.field_combo.blockSignals(True)
            self.y_combo.blockSignals(True)
            self.field_combo.clear()
            self.y_combo.clear()
            for key in keys:
                self.field_combo.addItem(display_name(key), key)
                self.y_combo.addItem(display_name(key), key)
            self.field_combo.blockSignals(False)
            self.y_combo.blockSignals(False)

            field_key = next((k for k in keys if k.lower() == "field"), keys[0])
            self.field_combo.setCurrentIndex(keys.index(field_key))

            preferred_y = next((k for k in keys if k.lower() in ("x", "y", "r")), None)
            if preferred_y is None:
                preferred_y = next((k for k in keys if k != field_key), field_key)
            self.y_combo.setCurrentIndex(keys.index(preferred_y))

            self._apply_selected_columns(columns, units)

        def _on_column_changed(self) -> None:
            if not self.data_path:
                return
            try:
                from FMRPreview import parse_numeric_columns

                columns, units, _header = parse_numeric_columns(self.data_path)
            except Exception:
                return
            self._apply_selected_columns(columns, units)

        def _apply_selected_columns(self, columns: Mapping[str, Sequence[float]], units: Mapping[str, str]) -> None:
            field_key = self.field_combo.currentData()
            y_key = self.y_combo.currentData()
            if not field_key or not y_key:
                return

            field = np.asarray(columns.get(field_key, []), dtype=float)
            y = np.asarray(columns.get(y_key, []), dtype=float)
            length = min(field.size, y.size)
            if length == 0:
                return
            field = field[:length]
            y = y[:length]

            unit = str(units.get(field_key, "") or "")
            if "koe" in unit.lower():
                field_koe = field
            elif "oe" in unit.lower():
                field_koe = field / 1000.0
            else:
                field_koe = field / 1000.0 if np.nanmax(np.abs(field)) > 200 else field

            self.field_values_koe = field_koe
            self.y_values = y
            self.y_label = str(y_key)
            self._update_range_suggestion()
            self.update_plot()

        def update_plot(self) -> None:
            self.plot_widget.setLabel("bottom", "Field (kOe)")
            base_label = self.y_label or "Signal"
            self.plot_widget.setLabel("left", base_label)

            if self.field_values_koe is not None and self.y_values is not None:
                x_data = self.field_values_koe
                y_data = self.y_values
                self.experimental_curve.setData(x_data, y_data)
            else:
                self.experimental_curve.setData([], [])

            start = float(self.model_field_start.value())
            stop = float(self.model_field_stop.value())
            points = int(self.model_field_points.value())
            if not (np.isfinite(start) and np.isfinite(stop)) or points < 2:
                self.model_curve.setData([], [])
                return
            if start == stop:
                stop = start + 1.0
            x_model = np.linspace(start, stop, points, dtype=float)

            self.plot_widget.setLabel("bottom", "Field (kOe)")

            params = self._params_dict()
            model = surface_impedance_absorbed_power(x_model, params["f"], params)
            model = np.asarray(model, dtype=float)
            if self.derivative_checkbox.isChecked():
                model = np.gradient(model, x_model)
            model = float(self.model_offset.value()) + float(self.model_scale.value()) * model
            self.model_curve.setData(x_model, model)

    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    window = FMRFittingWindow()
    window.resize(1400, 900)
    window.show()
    app.exec_()


if __name__ == "__main__":
    launch_ui()
