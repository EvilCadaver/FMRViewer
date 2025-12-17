from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING, Type

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except Exception:  # noqa: BLE001
    QtCore = None
    QtGui = None
    QtWidgets = None
    pg = None

if TYPE_CHECKING:
    from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox
    ModelClass = Type["BaseFMRModel"]
else:
    QCheckBox = object
    QDoubleSpinBox = object
    ModelClass = object

ArrayLike = np.ndarray | list[float] | tuple[float, ...]
ComplexArray = np.ndarray


def _as_float_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _as_complex_array(values: ArrayLike | ComplexArray) -> np.ndarray:
    return np.asarray(values, dtype=np.complex128)


def _broadcast_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        shape = np.broadcast(a, b).shape
    except ValueError as exc:
        raise ValueError("Field and frequency must be broadcastable to the same shape.") from exc
    return np.broadcast_to(a, shape).astype(float, copy=False), np.broadcast_to(b, shape).astype(float, copy=False)


def complex_from_components(real: ArrayLike, imag: ArrayLike) -> ComplexArray:
    real_arr = _as_float_array(real)
    imag_arr = _as_float_array(imag)
    real_arr, imag_arr = _broadcast_pair(real_arr, imag_arr)
    return real_arr + 1j * imag_arr


def complex_from_polar(magnitude: ArrayLike, phase_rad: ArrayLike) -> ComplexArray:
    magnitude_arr = _as_float_array(magnitude)
    phase_arr = _as_float_array(phase_rad)
    magnitude_arr, phase_arr = _broadcast_pair(magnitude_arr, phase_arr)
    return magnitude_arr * np.exp(1j * phase_arr)


def complex_magnitude(values: ComplexArray) -> np.ndarray:
    return np.abs(_as_complex_array(values))


def complex_phase(values: ComplexArray) -> np.ndarray:
    return np.angle(_as_complex_array(values))


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    default: float
    unit: str
    description: str = ""


DEFAULT_PARAMETER_SPECS: Sequence[ParameterSpec] = (
    ParameterSpec("Js", "Saturation polarization", 21.46, "kG", "Saturation polarization"),
    ParameterSpec("g", "Gyromagnetic factor", 2.088, "", "Dimensionless g-factor"),
    ParameterSpec("alpha", "Damping", 1.35e-3, "", "Dimensionless damping"),
    ParameterSpec("rho", "Resistivity", 9.7, "uOhm*cm", "Electrical resistivity"),
    ParameterSpec("f", "Frequency", 70.0, "GHz", "Microwave frequency"),
    ParameterSpec("gamma", "Gamma", 1.399611, "GHz/kOe", "User-supplied constant"),
)


@dataclass(frozen=True)
class ModelParameters:
    Js: float = 21.46
    g: float = 2.088
    alpha: float = 1.35e-3
    rho: float = 9.7
    f: float = 70.0
    gamma: float = 1.399611
    units: Dict[str, str] = field(
        default_factory=lambda: {
            "Js": "kG",
            "g": "",
            "alpha": "",
            "rho": "uOhm*cm",
            "f": "GHz",
            "gamma": "GHz/kOe",
        }
    )
    unit_system: str = "cgs"

    def to_dict(self) -> Dict[str, float]:
        return {
            "Js": self.Js,
            "g": self.g,
            "alpha": self.alpha,
            "rho": self.rho,
            "f": self.f,
            "gamma": self.gamma,
        }

    def with_overrides(self, overrides: Optional[Mapping[str, float]] = None) -> "ModelParameters":
        if not overrides:
            return self
        data = self.to_dict()
        data.update(overrides)
        return ModelParameters(**data, units=dict(self.units), unit_system=self.unit_system)


def parameter_specs() -> Sequence[ParameterSpec]:
    return DEFAULT_PARAMETER_SPECS


def cgs_to_si_parameters(params: ModelParameters) -> ModelParameters:
    # Optional helper: convert to SI if needed for specific equations.
    # Js: kG -> T (1 kG = 0.1 T)
    # rho: uOhm*cm -> ohm*m (1 uOhm*cm = 1e-8 ohm*m)
    # f: GHz -> Hz
    return ModelParameters(
        Js=params.Js * 0.1,
        g=params.g,
        alpha=params.alpha,
        rho=params.rho * 1e-8,
        f=params.f * 1e9,
        gamma=params.gamma,
        units={
            "Js": "T",
            "g": "",
            "alpha": "",
            "rho": "ohm*m",
            "f": "Hz",
            "gamma": params.units.get("gamma", ""),
        },
        unit_system="si",
    )


@dataclass(frozen=True)
class ModelInput:
    field: np.ndarray
    frequency: np.ndarray
    parameters: ModelParameters = field(default_factory=ModelParameters)
    metadata: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_values(
        cls,
        field_values: ArrayLike,
        frequency_values: ArrayLike,
        parameters: Optional[ModelParameters] = None,
        metadata: Optional[Mapping[str, float]] = None,
    ) -> "ModelInput":
        field_arr = _as_float_array(field_values)
        freq_arr = _as_float_array(frequency_values)
        field_arr, freq_arr = _broadcast_pair(field_arr, freq_arr)
        return cls(
            field=field_arr,
            frequency=freq_arr,
            parameters=parameters or ModelParameters(),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class ModelOutput:
    response: ComplexArray
    absorbed_power: np.ndarray
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


class BaseFMRModel:
    name = "base"

    def __init__(self, parameters: Optional[ModelParameters] = None):
        self.parameters: ModelParameters = parameters or ModelParameters()

    def compute_complex_response(self, model_input: ModelInput) -> ComplexArray:
        raise NotImplementedError("Define the model's complex response here.")

    def compute_absorbed_power(
        self, response: ComplexArray, model_input: ModelInput
    ) -> np.ndarray:
        raise NotImplementedError("Define the absorbed power calculation here.")

    def evaluate(self, model_input: ModelInput) -> ModelOutput:
        response = _as_complex_array(self.compute_complex_response(model_input))
        absorbed = _as_float_array(self.compute_absorbed_power(response, model_input))
        absorbed, _ = _broadcast_pair(absorbed, model_input.field)
        return ModelOutput(response=response, absorbed_power=absorbed)


class ModelOne(BaseFMRModel):
    name = "gurevich"

    def compute_complex_response(self, model_input: ModelInput) -> ComplexArray:
        params = model_input.parameters
        h_field = model_input.field
        ms = params.Js / (4.0 * np.pi)
        b_field = h_field + 4.0 * np.pi * ms
        omg = params.f / (params.gamma * params.g)
        numerator = -ms * (b_field - 1j * params.alpha * omg)
        denominator = (omg**2) - (h_field - 1j * params.alpha * omg) * (b_field - 1j * params.alpha * omg)
        return numerator / denominator

    def compute_absorbed_power(
        self, response: ComplexArray, model_input: ModelInput) -> np.ndarray:
        return np.imag(response) * (model_input.field**2)


class ModelTwo(BaseFMRModel):
    name = "kittel"

    def __init__(self, parameters: Optional[ModelParameters] = None, power_mode: str = "mu_r"):
        super().__init__(parameters)
        self.power_mode = power_mode

    def compute_complex_response(self, model_input: ModelInput) -> ComplexArray:
        params = model_input.parameters
        h_field = model_input.field
        b_field = h_field + params.Js
        omg = params.f / (params.gamma * params.g)
        numerator = (omg**2) - (b_field - 1j * params.alpha * omg) ** 2
        denominator = (omg**2) - (h_field - 1j * params.alpha * omg) * (b_field - 1j * params.alpha * omg)
        return numerator / denominator

    @staticmethod
    def _mu_components(response: ComplexArray) -> tuple[np.ndarray, np.ndarray]:
        real = np.real(response)
        imag = np.imag(response)
        magnitude = np.sqrt(real**2 + imag**2)
        mu_r = magnitude + imag
        mu_l = magnitude - imag
        return mu_r, mu_l

    def _select_mu_for_power(self, mu_r: np.ndarray, mu_l: np.ndarray) -> np.ndarray:
        return mu_l if self.power_mode == "mu_l" else mu_r

    def compute_absorbed_power(self, response: ComplexArray, model_input: ModelInput) -> np.ndarray:
        mu_r, mu_l = self._mu_components(response)
        mu_power = self._select_mu_for_power(mu_r, mu_l)
        return np.sqrt(model_input.parameters.rho * mu_power) * (model_input.field**2)

    def evaluate(self, model_input: ModelInput) -> ModelOutput:
        response = _as_complex_array(self.compute_complex_response(model_input))
        mu_r, mu_l = self._mu_components(response)
        mu_power = self._select_mu_for_power(mu_r, mu_l)
        absorbed = np.sqrt(model_input.parameters.rho * mu_power) * (model_input.field**2)
        absorbed, _ = _broadcast_pair(absorbed, model_input.field)
        extra = {"mu_R": mu_r, "mu_L": mu_l}
        return ModelOutput(response=response, absorbed_power=absorbed, extra=extra)


class ModelThree(BaseFMRModel):
    name = "model_three"

    def compute_complex_response(self, model_input: ModelInput) -> ComplexArray:
        # TODO: implement the complex response for model three.
        raise NotImplementedError("Fill in model three response.")

    def compute_absorbed_power(
        self, response: ComplexArray, model_input: ModelInput
    ) -> np.ndarray:
        # TODO: implement the absorbed power for model three.
        raise NotImplementedError("Fill in model three absorbed power.")


def compare_models(
    models: Iterable[BaseFMRModel], model_input: ModelInput
) -> Dict[str, ModelOutput]:
    results: Dict[str, ModelOutput] = {}
    for model in models:
        results[model.name] = model.evaluate(model_input)
    return results


MODEL_CLASSES = (ModelOne, ModelTwo, ModelThree)


if QtWidgets is not None:

    class FMRModelWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("FMR Model Viewer")

            self.plot_widget = pg.PlotWidget(background="w")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLabel("bottom", "Field (kOe)")
            self.plot_widget.setLabel("left", "Value")
            self.legend = None

            self.field_start = QtWidgets.QDoubleSpinBox()
            self.field_start.setRange(-1e6, 1e6)
            self.field_start.setDecimals(6)
            self.field_start.setValue(0.0)

            self.field_stop = QtWidgets.QDoubleSpinBox()
            self.field_stop.setRange(-1e6, 1e6)
            self.field_stop.setDecimals(6)
            self.field_stop.setValue(10.0)

            self.field_points = QtWidgets.QSpinBox()
            self.field_points.setRange(2, 1000000)
            self.field_points.setValue(1001)

            self.param_inputs: Dict[str, QDoubleSpinBox] = {}
            self.param_group = self._build_parameter_controls()

            self.plot_type_combo = QtWidgets.QComboBox()
            self.plot_type_combo.addItem("Absorbed power", "absorbed_power")
            self.plot_type_combo.addItem("Response real", "response_real")
            self.plot_type_combo.addItem("Response imag", "response_imag")
            self.plot_type_combo.addItem("Response magnitude", "response_mag")
            self.plot_type_combo.addItem("Response phase (rad)", "response_phase")
            self.log_scale_checkbox = QtWidgets.QCheckBox("Log10 scale")

            self.model_checks: Dict[ModelClass, QCheckBox] = {}
            self.kittel_power_combo = None
            self.model_group = self._build_model_controls()

            self.update_button = QtWidgets.QPushButton("Update plot")
            self.update_button.clicked.connect(self.update_plot)

            controls_layout = QtWidgets.QVBoxLayout()
            controls_layout.addWidget(self._build_field_controls())
            controls_layout.addWidget(self.param_group)
            controls_layout.addWidget(self._build_plot_controls())
            controls_layout.addWidget(self.model_group)
            controls_layout.addWidget(self.update_button)
            controls_layout.addStretch()

            controls_widget = QtWidgets.QWidget()
            controls_widget.setLayout(controls_layout)

            main_layout = QtWidgets.QHBoxLayout()
            main_layout.addWidget(self.plot_widget, 1)
            main_layout.addWidget(controls_widget)

            container = QtWidgets.QWidget()
            container.setLayout(main_layout)
            self.setCentralWidget(container)
            self.statusBar().showMessage("Ready")

        def _build_field_controls(self):
            group = QtWidgets.QGroupBox("Field sweep")
            layout = QtWidgets.QFormLayout()
            layout.addRow("Start (kOe)", self.field_start)
            layout.addRow("Stop (kOe)", self.field_stop)
            layout.addRow("Points", self.field_points)
            group.setLayout(layout)
            return group

        def _build_parameter_controls(self):
            group = QtWidgets.QGroupBox("Model parameters")
            layout = QtWidgets.QFormLayout()
            layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            for spec in parameter_specs():
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(-1e12, 1e12)
                spin.setDecimals(6)
                spin.setSingleStep(self._parameter_step(spec.default))
                spin.setValue(spec.default)
                layout.addRow(self._label_with_unit(spec), spin)
                self.param_inputs[spec.key] = spin
            group.setLayout(layout)
            return group

        def _build_plot_controls(self):
            group = QtWidgets.QGroupBox("Plot")
            layout = QtWidgets.QFormLayout()
            layout.addRow("Plot type", self.plot_type_combo)
            layout.addRow(self.log_scale_checkbox)
            group.setLayout(layout)
            return group

        def _build_model_controls(self):
            group = QtWidgets.QGroupBox("Models")
            layout = QtWidgets.QVBoxLayout()
            for model_cls in MODEL_CLASSES:
                label = model_cls.name.replace("_", " ").title()
                checkbox = QtWidgets.QCheckBox(label)
                checkbox.setChecked(True)
                if model_cls is ModelTwo:
                    row = QtWidgets.QHBoxLayout()
                    row.addWidget(checkbox)
                    row.addStretch()
                    combo = QtWidgets.QComboBox()
                    combo.addItem("mu_R", "mu_r")
                    combo.addItem("mu_L", "mu_l")
                    combo.setToolTip("Absorbed power uses selected mu")
                    row.addWidget(combo)
                    layout.addLayout(row)
                    self.kittel_power_combo = combo
                else:
                    layout.addWidget(checkbox)
                self.model_checks[model_cls] = checkbox
            layout.addStretch()
            group.setLayout(layout)
            return group

        @staticmethod
        def _parameter_step(value: float) -> float:
            magnitude = abs(value)
            if magnitude >= 100:
                return 1.0
            if magnitude >= 10:
                return 0.1
            if magnitude >= 1:
                return 0.01
            if magnitude >= 0.1:
                return 0.001
            return 0.0001

        @staticmethod
        def _label_with_unit(spec: ParameterSpec) -> str:
            if spec.unit:
                return f"{spec.label} ({spec.unit})"
            return spec.label

        def _collect_parameters(self) -> ModelParameters:
            overrides = {key: spin.value() for key, spin in self.param_inputs.items()}
            return ModelParameters().with_overrides(overrides)

        def update_plot(self):
            selected = [cls for cls, chk in self.model_checks.items() if chk.isChecked()]
            if not selected:
                self.statusBar().showMessage("Select at least one model.")
                return

            start = self.field_start.value()
            stop = self.field_stop.value()
            points = self.field_points.value()
            if points < 2:
                self.statusBar().showMessage("Field points must be >= 2.")
                return
            field = np.linspace(start, stop, points)
            params = self._collect_parameters()
            frequency = np.full_like(field, params.f)
            model_input = ModelInput.from_values(field, frequency, params)

            self.plot_widget.clear()
            self.legend = self.plot_widget.addLegend()

            plot_mode = self.plot_type_combo.currentData()
            y_label = self._label_for_mode(plot_mode)
            if self.log_scale_checkbox.isChecked():
                y_label = f"log10({y_label})"
            self.plot_widget.setLabel("bottom", "Field (kOe)")
            self.plot_widget.setLabel("left", y_label)

            errors = []
            colors = [
                (220, 70, 70),
                (60, 130, 220),
                (60, 170, 140),
            ]
            for idx, model_cls in enumerate(selected):
                if model_cls is ModelTwo:
                    power_mode = "mu_r"
                    if self.kittel_power_combo is not None:
                        power_mode = self.kittel_power_combo.currentData() or "mu_r"
                    model = model_cls(parameters=params, power_mode=power_mode)
                else:
                    model = model_cls(parameters=params)
                try:
                    output = model.evaluate(model_input)
                except NotImplementedError as exc:
                    errors.append(f"{model_cls.name}: {exc}")
                    continue
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{model_cls.name}: {exc}")
                    continue

                y_values = self._series_from_output(plot_mode, output)
                if self.log_scale_checkbox.isChecked():
                    y_values = self._apply_log10(y_values)
                pen = pg.mkPen(color=colors[idx % len(colors)], width=2)
                label = model_cls.name.replace("_", " ").title()
                self.plot_widget.plot(field, y_values, pen=pen, name=label)

            if errors:
                self.statusBar().showMessage(" | ".join(errors)[:300])
            else:
                self.statusBar().showMessage("Plot updated")

        @staticmethod
        def _series_from_output(mode: str, output: ModelOutput) -> np.ndarray:
            if mode == "absorbed_power":
                return output.absorbed_power
            response = output.response
            if mode == "response_real":
                return np.real(response)
            if mode == "response_imag":
                return np.imag(response)
            if mode == "response_mag":
                return complex_magnitude(response)
            if mode == "response_phase":
                return complex_phase(response)
            return output.absorbed_power

        @staticmethod
        def _label_for_mode(mode: str) -> str:
            labels = {
                "absorbed_power": "Absorbed power",
                "response_real": "Response real",
                "response_imag": "Response imag",
                "response_mag": "Response magnitude",
                "response_phase": "Response phase (rad)",
            }
            return labels.get(mode, "Value")

        @staticmethod
        def _apply_log10(values: np.ndarray) -> np.ndarray:
            safe = np.abs(values)
            safe = np.where(safe <= 0, np.finfo(float).tiny, safe)
            return np.log10(safe)


def launch_ui():
    if QtWidgets is None or pg is None:
        raise RuntimeError("PySide6 and pyqtgraph are required to launch the UI.")
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.black)
    palette.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.AlternateBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
    palette.setColor(QtGui.QPalette.Button, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.black)
    palette.setColor(QtGui.QPalette.Highlight, QtCore.Qt.lightGray)
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)
    window = FMRModelWindow()
    window.resize(1400, 900)
    window.show()
    app.exec()


if __name__ == "__main__":
    launch_ui()
