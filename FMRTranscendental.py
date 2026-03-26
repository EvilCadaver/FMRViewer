from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget
else:
    QWidget = object

try:
    from PySide6 import QtWidgets
    import pyqtgraph as pg
except Exception:  # noqa: BLE001
    QtWidgets = None
    pg = None


def _float_range(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive float range with stable endpoint handling."""
    if step <= 0:
        raise ValueError("Step must be > 0.")
    if stop < start:
        raise ValueError("Stop must be >= start.")

    # Include endpoint (if it falls on the grid within floating tolerance).
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    values = start + step * np.arange(count, dtype=float)

    if values.size == 0:
        return np.asarray([start], dtype=float)
    if values[-1] < stop - 1e-12:
        values = np.append(values, stop)
    else:
        values[-1] = stop
    return values


def _pick_sintheta(roots: np.ndarray, eps: float = 1e-7) -> float:
    """Pick the smallest nonnegative physical root for sin(theta)."""
    real_roots = [
        root.real
        for root in roots
        if abs(root.imag) <= eps and -1.0 - eps <= root.real <= 1.0 + eps
    ]
    candidates = [0.0 if abs(val) <= eps else val for val in real_roots if val >= -eps]
    candidates = [None if val > 1 else val for val in candidates]
    if not candidates:
        return float("nan")
    return float(min(candidates))


def theta_from_h_phi(h_koe: float, phi_deg: float, hk_koe: float, ms_kgauss: float) -> float:
    """Return theta in degrees for one H,phi pair."""
    phi = np.radians(phi_deg)
    a_term = hk_koe * np.sin(2.0 * phi)
    b_term = 4.0 * np.pi * ms_kgauss + hk_koe * np.cos(2.0 * phi)
    c_term = 0.5 * hk_koe * np.sin(2.0 * phi)

    # (A^2+B^2)x^4 + 2AHx^3 - (2AC+B^2)x^2 - 2HCx + C^2 = 0, x = sin(theta)
    coeffs = [
        a_term**2 + b_term**2,
        2.0 * a_term * h_koe,
        -(2.0 * a_term * c_term + b_term**2),
        -2.0 * h_koe * c_term,
        c_term**2,
    ]
    sin_theta = _pick_sintheta(np.roots(coeffs))
    if np.isnan(sin_theta):
        return float("nan")
    return float(np.degrees(np.arcsin(sin_theta)))


def generate_curves(
    hk_koe: float,
    ms_kgauss: float,
    h_start: float,
    h_stop: float,
    h_step: float,
    phi_start: float,
    phi_stop: float,
    phi_step: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray]]]:
    h_values = _float_range(h_start, h_stop, h_step)
    phi_values = _float_range(phi_start, phi_stop, phi_step)

    curves: list[tuple[float, np.ndarray]] = []
    for phi_deg in phi_values:
        theta_values = np.asarray(
            [theta_from_h_phi(h_val, phi_deg, hk_koe, ms_kgauss) for h_val in h_values],
            dtype=float,
        )
        curves.append((float(phi_deg), theta_values))

    return h_values, phi_values, curves


if QtWidgets is not None and pg is not None:

    class TranscendentalWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("FMR Transcendental Theta(H)")

            self.plot_widget = pg.PlotWidget(background="w")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLabel("bottom", "H (kOe)")
            self.plot_widget.setLabel("left", "theta (deg)")
            self.legend = None

            self.hk_input = QtWidgets.QDoubleSpinBox()
            self.hk_input.setRange(-1e6, 1e6)
            self.hk_input.setDecimals(6)
            self.hk_input.setValue(5.0)

            self.ms_input = QtWidgets.QDoubleSpinBox()
            self.ms_input.setRange(-1e6, 1e6)
            self.ms_input.setDecimals(6)
            self.ms_input.setValue(8.0)

            self.h_start = QtWidgets.QDoubleSpinBox()
            self.h_start.setRange(-1e6, 1e6)
            self.h_start.setDecimals(6)
            self.h_start.setValue(0.0)

            self.h_stop = QtWidgets.QDoubleSpinBox()
            self.h_stop.setRange(-1e6, 1e6)
            self.h_stop.setDecimals(6)
            self.h_stop.setValue(20.0)

            self.h_step = QtWidgets.QDoubleSpinBox()
            self.h_step.setRange(1e-9, 1e6)
            self.h_step.setDecimals(6)
            self.h_step.setValue(0.05)

            self.phi_start = QtWidgets.QDoubleSpinBox()
            self.phi_start.setRange(-360.0, 360.0)
            self.phi_start.setDecimals(6)
            self.phi_start.setValue(0.0)

            self.phi_stop = QtWidgets.QDoubleSpinBox()
            self.phi_stop.setRange(-360.0, 360.0)
            self.phi_stop.setDecimals(6)
            self.phi_stop.setValue(90.0)

            self.phi_step = QtWidgets.QDoubleSpinBox()
            self.phi_step.setRange(1e-9, 360.0)
            self.phi_step.setDecimals(6)
            self.phi_step.setValue(10.0)

            self.update_button = QtWidgets.QPushButton("Update plot")
            self.update_button.clicked.connect(self.update_plot)

            controls = self._build_controls()

            layout = QtWidgets.QHBoxLayout()
            layout.addWidget(self.plot_widget, 1)
            layout.addWidget(controls)

            container = QtWidgets.QWidget()
            container.setLayout(layout)
            self.setCentralWidget(container)
            self.statusBar().showMessage("Ready")

            self.update_plot()

        def _build_controls(self) -> QWidget:
            params_group = QtWidgets.QGroupBox("Material")
            params_form = QtWidgets.QFormLayout()
            params_form.addRow("Hk (kOe)", self.hk_input)
            params_form.addRow("Ms (kG)", self.ms_input)
            params_group.setLayout(params_form)

            h_group = QtWidgets.QGroupBox("H Sweep")
            h_form = QtWidgets.QFormLayout()
            h_form.addRow("Start", self.h_start)
            h_form.addRow("Stop", self.h_stop)
            h_form.addRow("Step", self.h_step)
            h_group.setLayout(h_form)

            phi_group = QtWidgets.QGroupBox("phi Sweep (deg)")
            phi_form = QtWidgets.QFormLayout()
            phi_form.addRow("Start", self.phi_start)
            phi_form.addRow("Stop", self.phi_stop)
            phi_form.addRow("Step", self.phi_step)
            phi_group.setLayout(phi_form)

            panel_layout = QtWidgets.QVBoxLayout()
            panel_layout.addWidget(params_group)
            panel_layout.addWidget(h_group)
            panel_layout.addWidget(phi_group)
            panel_layout.addWidget(self.update_button)
            panel_layout.addStretch()

            panel = QtWidgets.QWidget()
            panel.setLayout(panel_layout)
            panel.setMaximumWidth(320)
            return panel

        def update_plot(self) -> None:
            try:
                h_values, phi_values, curves = generate_curves(
                    hk_koe=float(self.hk_input.value()),
                    ms_kgauss=float(self.ms_input.value()),
                    h_start=float(self.h_start.value()),
                    h_stop=float(self.h_stop.value()),
                    h_step=float(self.h_step.value()),
                    phi_start=float(self.phi_start.value()),
                    phi_stop=float(self.phi_stop.value()),
                    phi_step=float(self.phi_step.value()),
                )
            except ValueError as exc:
                self.statusBar().showMessage(str(exc))
                return

            h_points = h_values.size
            phi_points = phi_values.size
            total_points = h_points * phi_points
            if total_points > 2_000_000:
                self.statusBar().showMessage(
                    f"Too many points ({total_points}). Increase steps or reduce ranges."
                )
                return

            self.plot_widget.clear()
            self.legend = self.plot_widget.addLegend()

            palette = [
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
            ]

            for idx, (phi_deg, theta_values) in enumerate(curves):
                pen = pg.mkPen(color=palette[idx % len(palette)], width=2)
                self.plot_widget.plot(h_values, theta_values, pen=pen, name=f"phi={phi_deg:g} deg")

            self.statusBar().showMessage(
                f"Plotted {phi_points} curves ({h_points} H points each)."
            )



def launch_ui() -> None:
    if QtWidgets is None or pg is None:
        raise RuntimeError("PySide6 and pyqtgraph are required to launch the UI.")

    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    window = TranscendentalWindow()
    window.resize(1400, 900)
    window.show()
    app.exec()


if __name__ == "__main__":
    launch_ui()


