"""
Name: Circuit Simulation Solver
Author: Akarsh Duddu
Email: aduddu3@gatech.edu
"""

from collections.abc import Callable
import numpy as np

from functools import partial
import matplotlib.pyplot as plt


def forward_euler(
    xp,
    dydx: Callable[[float, float], float],
    span: tuple[float, float],
    y0: np.ndarray,
    first_step: float,
):
    """Forward Euler method of approximating ordinary differential equations (ODEs)."""
    # Builtin range function does not support floating-point step
    curr = span[0]
    inputs = []
    while curr < span[1]:
        inputs.append(curr)
        curr += first_step

    step = first_step
    outputs = [None for _ in inputs]
    outputs[0] = y0

    for i in range(1, len(inputs)):
        # y_new = y + dy/dx * delta x

        dydt_vector = dydx(inputs[i - 1], outputs[i - 1])
        outputs[i] = [
            outputs[i - 1][j] + dydt_vector[j] * step
            for j in range(len(y0))
        ]

    return (inputs, outputs)


def rc(t: float, Vc: float, R: float, C: float, Vs_func: Callable[[float], float]) -> float:
    tau = R * C
    Vs = Vs_func(t)  # Get the current source voltage (allows for any input!)
    return [(Vs - Vc[0]) / tau] # dV/dt


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

    time, voltage = forward_euler(np, dVdt, (0, t_max), [V_C_initial], t_max / 1000)

    plt.figure(figsize=(8, 5))
    plt.plot([t * 1e3 for t in time], voltage, label="Numerical Solution")
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

    y0 = np.array([10, 5]) # Initial prey, initial predators

    dydt = partial(lotka_volterra, a=a, b=b, c=c, d=d)
    # dydt = lambda t, y: lotka_volterra(t, y, a, b, c, d)

    # Solve
    t, sol = forward_euler(np, dydt, (0, 30), y0, 1)
    prey, predators = zip(*sol)
    
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
    t, sol = forward_euler(np, dVdt, (0, t_max), y0, t_max/1000)

    plt.plot([i * 1000 for i in t], [i[0] for i in sol])
    plt.xlabel("Time (ms)")
    plt.ylabel("Capacitor Voltage (V)")
    plt.title("Series RLC Step Response (Using solve_ivp2)")
    # plt.ylim((4, 6))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    display_lv()
