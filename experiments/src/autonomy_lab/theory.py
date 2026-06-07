import math


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def x_star_sigmoid(a: float, C: float, k: float, x_c: float) -> float | None:
    """Interior equilibrium for p(x)=sigmoid(k*(x-x_c))."""
    if a <= 0 or a >= C:
        return None
    return x_c + (1.0 / k) * math.log(a / (C - a))


def delta_c_general(C: float, x_star: float, p_prime: float) -> float:
    denom = 2.0 * C * x_star * (1.0 - x_star) * p_prime
    if denom <= 0:
        return float('inf')
    return math.pi / denom


def delta_c_sigmoid(a: float, C: float, k: float, x_star: float) -> float:
    """Critical delay for sigmoid response, using p(x*)=a/C."""
    if a <= 0 or a >= C or not (0.0 < x_star < 1.0):
        return float('inf')
    p_prime = k * (a / C) * (1.0 - a / C)
    return delta_c_general(C, x_star, p_prime)
