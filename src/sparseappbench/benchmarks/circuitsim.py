"""
Name: Circuit Simulation Solver
Author: Akarsh Duddu
Email: aduddu3@gatech.edu
"""

from collections.abc import Callable
from functools import partial

import numpy as np

import matplotlib.pyplot as plt


def forward_euler(
    dydx: Callable[[float, float], float], span: tuple[float, float], y0: np.ndarray,
    first_step: float | None = None, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Forward Euler method of approximating ordinary differential equations (ODEs)."""
    if first_step is None:
        inputs = np.linspace(span[0], span[1], 1000)
        step = inputs[1] - inputs[0]
    else:
        inputs = np.arange(span[0], span[1], first_step)
        step = first_step
    outputs = np.zeros((len(inputs), len(y0)))
    outputs[0] = y0

    for i in range(1, inputs.shape[0]):
        # y_new = y + dy/dx * delta x
        outputs[i] = (
            outputs[i - 1] + np.array(dydx(inputs[i - 1], outputs[i - 1])) * step
        )

    return (inputs, outputs)


def rc(t: float, Vc: float, R: float, C: float, Vs_func: Callable[[float], float]) -> float:
    tau = R * C
    Vs = Vs_func(t)  # Get the current source voltage (allows for any input!)
    return (Vs - Vc) / tau # dV/dt


def rlc(t: float, state: np.ndarray, R: float, L: float, C: float, Vs_func: Callable[[float], float]) -> tuple[float, float]:
    Vc = state[0]
    dVc = state[1]  # dx1/dt = x2

    Vs = Vs_func(t)

    d2Vc = (Vs - Vc - R * C * dVc) / (L * C)  # dx2/dt = (Vs - x1 - RCx2) / LC
    return (dVc, d2Vc)


def lotka_volterra(t: float, state: np.ndarray, a: float, b: float, c: float, d: float) -> tuple[float, float]:
    x, y = state
    dxdt = a * x - b * x * y
    dydt = d * x * y - c * y
    return (dxdt, dydt)


def step_input(t: float) -> float:
    """A simple 5V step input starting at t=0."""
    return 5.0 if t >= 0 else 0.0


def display_rc() -> None:
    R = 10e3  # 10 kΩ
    C = 1e-6  # 1 µF
    t_max = 5 * R * C
    V_C_initial = 0.0

    dVdt = partial(rc, R=R, C=C, Vs_func=step_input)
    # dVdt = lambda t, Vc: rc_ode(t, Vc, R, C, step_input)

    time, voltage = forward_euler(dVdt, (0, t_max), [V_C_initial])

    plt.figure(figsize=(8, 5))
    plt.plot(time * 1e3, voltage, label="Numerical Solution")
    plt.title("RC Circuit Charging - Numerical Solution")
    plt.xlabel("Time (ms)")
    plt.ylabel("Capacitor Voltage (V)")
    plt.grid(True)
    plt.legend()
    plt.show()


def display_lv() -> None:
    # Parameters
    a = 1.1  # prey birth rate
    b = 0.4  # predation rate
    c = 0.4  # predator death rate
    d = 0.1  # predator reproduction rate

    y0 = np.ndarray([10, 5]) # Initial prey, initial predators

    dydt = partial(lotka_volterra, a=a, b=b, c=c, d=d)
    # dydt = lambda t, y: lotka_volterra(t, y, a, b, c, d)

    # Solve
    t, sol = forward_euler(dydt, (0, 30), y0)
    prey = sol[:, 0]
    predators = sol[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(t, prey, label="Prey")
    plt.plot(t, predators, label="Predators")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Lotka-Volterra Predator-Prey Dynamics")
    plt.grid(True)
    plt.legend()
    plt.show()


def display_rlc() -> None:
    R = 100
    L = 10e-3
    C = 1e-7
    t_max = 10 * L / R

    y0 = np.array([0.0, 0.0]) # y0[0] is Vc, y0[1] is dVc/dt
    dVdt = partial(rlc, R=R, L=L, C=C, Vs_func=step_input)
    # dVdt = lambda t, y: rlc(t, y, R, L, C, step_input)
    t, sol = forward_euler(dVdt, (0, t_max), y0)

    plt.plot(t * 1000, sol[:, 0])
    plt.xlabel("Time (ms)")
    plt.ylabel("Capacitor Voltage (V)")
    plt.title("Series RLC Step Response (Using solve_ivp2)")
    # plt.ylim((4, 6))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    display_rc()
