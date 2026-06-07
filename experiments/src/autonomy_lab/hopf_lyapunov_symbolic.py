"""
Symbolic proof that Re(c_1(0)) < 0 for the delayed replicator equation:

    dx/dt = x(t)(1-x(t))[a - C*p(x(t-Delta))]

where p(x) = sigmoid(k(x - x_c)).

This script computes the first Lyapunov coefficient via center manifold
reduction (Hassard-Kazarinoff-Wan) and proves algebraically that the Hopf
bifurcation at Delta_c is supercritical for ALL admissible parameters.

The proof structure:
  1. Rationalize the center manifold computation (no complex denominators)
  2. Obtain Re(c_1) = C*k*rho*N / [5*alpha_0*(4+pi^2)]
  3. Factor N = (rho-1)*B where B is a positive definite quadratic form
  4. Conclude Re(c_1) < 0 universally
"""

import sympy as sp
from sympy import symbols, Rational, pi, simplify, factor, expand, cancel
import numpy as np


def compute_Re_c1_symbolic():
    """
    Compute Re(c_1) symbolically for the general case (arbitrary rho).
    Returns the symbolic expression and its factored numerator.
    """
    k, C, rho = symbols('k C rho', positive=True, real=True)
    a0 = symbols('alpha_0', positive=True, real=True)
    a1 = symbols('alpha_1', real=True)

    # Sigmoid derivatives at rho = a/C
    q = rho * (1 - rho)
    p1 = k * q
    p2 = k**2 * q * (1 - 2*rho)
    p3 = k**3 * q * (1 - 6*q)

    sigma = pi / 2
    S = 1 + sigma**2  # = 1 + pi^2/4

    # Linear coefficient
    b = C * a0 * p1
    omega = b

    # D_bar = (1 - i*sigma)/(1+sigma^2): real and imaginary parts
    D_r = sp.Integer(1) / S
    D_i = -sigma / S

    # === Quadratic nonlinearity on center manifold ===
    f20_re = C * a0 * p2 / 2
    f20_im = C * a1 * p1
    f11_re = -C * a0 * p2
    f02_re = C * a0 * p2 / 2
    f02_im = -C * a1 * p1

    # g_20 = 2*D_bar*f_20
    g20_re = 2*(D_r*f20_re - D_i*f20_im)
    g20_im = 2*(D_r*f20_im + D_i*f20_re)

    # g_11 = D_bar*f_11 (f_11 is real)
    g11_re = D_r * f11_re
    g11_im = D_i * f11_re

    # g_02 = 2*D_bar*f_02
    g02_re = 2*(D_r*f02_re - D_i*f02_im)
    g02_im = 2*(D_r*f02_im + D_i*f02_re)

    # conj(g_02)
    g02bar_re = g02_re
    g02bar_im = -g02_im

    # === E_1 = 2*f_20 / [b*(2i-1)] ===
    # 1/(2i-1) = (-1-2i)/5
    E1_re = 2*(-f20_re + 2*f20_im) / (5*b)
    E1_im = 2*(-f20_im - 2*f20_re) / (5*b)

    # === W_20(0) = i*g_20/omega + i*conj(g_02)/(3*omega) + E_1 ===
    W20_0_re = simplify(-g20_im/omega + (-g02bar_im)/(3*omega) + E1_re)
    W20_0_im = simplify(g20_re/omega + g02bar_re/(3*omega) + E1_im)

    # === W_20(-tau) = g_20/omega - conj(g_02)/(3*omega) - E_1 ===
    W20_tau_re = simplify(g20_re/omega - g02bar_re/(3*omega) - E1_re)
    W20_tau_im = simplify(g20_im/omega - g02bar_im/(3*omega) - E1_im)

    # === W_11 (non-zero since p2 != 0) ===
    E2 = f11_re / b
    W11_0_re = simplify(2*g11_im/omega + E2)
    W11_tau_re = simplify(-2*g11_re/omega + E2)

    # === Cubic on center manifold ===
    f3_re = -Rational(1, 2)*C*a1*p2
    f3_im = -C*p1 + Rational(1, 2)*C*a0*p3

    # === W corrections ===
    # piece1 = -i*C*alpha_1*p1 * [W_20(0)/2 - W_11(0)]
    term1_re = W20_0_re/2 - W11_0_re
    term1_im = W20_0_im/2
    piece1_re = C*a1*p1 * term1_im
    piece1_im = -C*a1*p1 * term1_re

    # piece2 = -C*alpha_1*p1 * [W_20(-tau)/2 + W_11(-tau)]
    term2_re = W20_tau_re/2 + W11_tau_re
    term2_im = W20_tau_im/2
    piece2_re = -C*a1*p1 * term2_re
    piece2_im = -C*a1*p1 * term2_im

    # piece3 = -i*C*alpha_0*p2 * [W_20(-tau)/2 - W_11(-tau)]
    term3_re = W20_tau_re/2 - W11_tau_re
    term3_im = W20_tau_im/2
    piece3_re = C*a0*p2 * term3_im
    piece3_im = -C*a0*p2 * term3_re

    CW_re = simplify(piece1_re + piece2_re + piece3_re)
    CW_im = simplify(piece1_im + piece2_im + piece3_im)

    # === Total f_21 and Re(g_21) ===
    f21_re = simplify(f3_re + CW_re)
    f21_im = simplify(f3_im + CW_im)

    # Re(g_21) = 2*(D_r*f21_re - D_i*f21_im) = 2*(f21_re + sigma*f21_im)/S
    Re_g21 = simplify(2*(f21_re + sigma * f21_im) / S)

    # === Full Re(c_1) ===
    # Re[i*(g20*g11)/(2*omega)] = -(g20_re*g11_im + g20_im*g11_re)/(2*omega)
    Re_first_term = -(g20_re*g11_im + g20_im*g11_re)/(2*omega)

    Re_c1 = simplify(Re_first_term + Re_g21/2)

    return Re_c1, (k, C, rho, a0, a1)


def verify_bvp_symbolic():
    """
    Symbolically verify that W_20 and W_11 satisfy their boundary conditions
    for ALL parameter values (not just numerically for specific ones).
    """
    k, C = symbols('k C', positive=True, real=True)
    rho = symbols('rho', positive=True, real=True)
    a0 = symbols('alpha_0', positive=True, real=True)
    a1 = symbols('alpha_1', real=True)

    sigma = pi / 2
    S = 1 + sigma**2
    q = rho * (1 - rho)
    p1 = k * q
    p2 = k**2 * q * (1 - 2*rho)

    b = C * a0 * p1
    omega = b
    D_r = sp.Integer(1) / S
    D_i = -sigma / S

    f20_re = C * a0 * p2 / 2
    f20_im = C * a1 * p1
    f11_re = -C * a0 * p2

    g20_re = 2*(D_r*f20_re - D_i*f20_im)
    g20_im = 2*(D_r*f20_im + D_i*f20_re)
    g02_re = 2*(D_r*f20_re + D_i*f20_im)
    g02_im = 2*(-D_r*f20_im + D_i*f20_re)
    g02bar_re = g02_re
    g02bar_im = -g02_im
    g11_re = D_r * f11_re

    E1_re = 2*(-f20_re + 2*f20_im) / (5*b)
    E1_im = 2*(-f20_im - 2*f20_re) / (5*b)
    E2 = f11_re / b

    W20_0_re = simplify(-g20_im/omega + (-g02bar_im)/(3*omega) + E1_re)
    W20_0_im = simplify(g20_re/omega + g02bar_re/(3*omega) + E1_im)
    W20_tau_re = simplify(g20_re/omega - g02bar_re/(3*omega) - E1_re)
    W20_tau_im = simplify(g20_im/omega - g02bar_im/(3*omega) - E1_im)
    W11_tau_re = simplify(-2*g11_re/omega + E2)

    # BVP 1: 2iω·W_20(0) + b·W_20(-τ) = 2f_20 - g_20 - conj(g_02)
    LHS_re = -2*omega*W20_0_im + b*W20_tau_re
    LHS_im = 2*omega*W20_0_re + b*W20_tau_im
    RHS_re = 2*f20_re - g20_re - g02bar_re
    RHS_im = 2*f20_im - g20_im - g02bar_im

    err_20_re = simplify(expand(LHS_re - RHS_re))
    err_20_im = simplify(expand(LHS_im - RHS_im))

    # BVP 2: b·W_11(-τ) = f_11·(1 - D̄ - D) = f_11·(1 - 2/S)
    coeff = 1 - 2*D_r
    err_11 = simplify(expand(b*W11_tau_re - f11_re*coeff))

    print("\n" + "=" * 72)
    print("  SYMBOLIC BVP VERIFICATION")
    print("=" * 72)
    print(f"\n  W_20 boundary condition: 2iω·W_20(0) + b·W_20(-τ) = 2f_20 - g_20 - ḡ_02")
    print(f"    Re error = {err_20_re}")
    print(f"    Im error = {err_20_im}")
    assert err_20_re == 0 and err_20_im == 0, "W_20 BVP FAILED"
    print(f"    ✓ Satisfied identically for all parameters")

    print(f"\n  W_11 boundary condition: b·W_11(-τ) = f_11·(1 - D̄ - D)")
    print(f"    Error = {err_11}")
    assert err_11 == 0, "W_11 BVP FAILED"
    print(f"    ✓ Satisfied identically for all parameters")


def prove_sign():
    """Algebraic proof that Re(c_1) < 0 for all admissible parameters."""
    Re_c1, (k, C, rho, a0, a1) = compute_Re_c1_symbolic()

    print("=" * 72)
    print("  ALGEBRAIC PROOF: Re(c_1(0)) < 0 UNIVERSALLY")
    print("=" * 72)

    # Extract numerator and denominator
    n, d = sp.fraction(cancel(Re_c1))
    n = expand(n)
    d = factor(d)
    print(f"\n  Re(c_1) = Numerator / Denominator")
    print(f"  Denominator = {d}  [> 0 since alpha_0, 4+pi^2 > 0]")

    # Factor (rho-1) from numerator: n = C*k*rho*(rho-1)*bracket
    bracket = simplify(n / (C * k * rho * (rho - 1)))
    print(f"\n  Numerator = C*k*rho*(rho-1) * B")
    print(f"  where B = {bracket}")

    # The known factored structure from manual computation:
    # B = 2*Q(rho)*(a0*k)^2 - (7*pi-8)*(2*rho-1)*(a0*k)*a1 + 2*(3*pi-2)*a1^2 + 10*pi*a0
    # where Q(rho) = -(7*pi-8)*rho*(1-rho) + (3*pi-2)

    a_c = 7*pi - 8
    b_c = 3*pi - 2
    Q_expr = -a_c * rho * (1 - rho) + b_c

    # Verify structure by substitution
    B_expected = (2*Q_expr*(a0*k)**2
                  - a_c*(2*rho-1)*(a0*k)*a1
                  + 2*b_c*a1**2
                  + 10*pi*a0)
    diff = simplify(expand(bracket - B_expected))
    assert diff == 0, f"Structure mismatch: {diff}"

    print(f"\n  Verified: B = 2*Q*(alpha_0*k)^2 - (7pi-8)*(2rho-1)*(alpha_0*k)*alpha_1")
    print(f"               + 2*(3pi-2)*alpha_1^2 + 10*pi*alpha_0")
    print(f"  with Q(rho) = -(7pi-8)*rho*(1-rho) + (3pi-2)")

    # Step 1: Q(rho) > 0
    print(f"\n  Step 1: Q(rho) >= 5*pi/4 > 0")
    print(f"    Since rho*(1-rho) <= 1/4, Q >= -(7pi-8)/4 + (3pi-2) = 5pi/4")
    Q_min = simplify(-a_c/4 + b_c)
    print(f"    = {Q_min} = {float(Q_min):.4f} > 0  ✓")

    # Step 2: Quadratic form positive definite
    print(f"\n  Step 2: Quadratic form in (alpha_0*k, alpha_1) is positive definite")
    print(f"    A = 2*Q, C = 2*(3pi-2), B_cross = -(7pi-8)*(2rho-1)")
    print(f"    Need: 4AC > B_cross^2, i.e., 16*Q*(3pi-2) > (7pi-8)^2*(2rho-1)^2")
    print(f"    With u = rho*(1-rho), (2rho-1)^2 = 1-4u:")
    print(f"    LHS - RHS = [-20pi(7pi-8)]u + [95pi^2-80pi]")
    print(f"    Worst at u=1/4: = 20pi(3pi-2) = {float(20*pi*b_c):.1f} > 0  ✓")

    # Step 3: Bonus term
    print(f"\n  Step 3: Additional term 10*pi*alpha_0 > 0  ✓")

    # Conclusion
    print(f"\n  Conclusion:")
    print(f"    B > 0 for all admissible parameters.")
    print(f"    N = C*k*rho*(rho-1)*B, with (rho-1) < 0.")
    print(f"    Re(c_1) = N / [5*alpha_0*(4+pi^2)] < 0.")
    print(f"\n  QED: The Hopf bifurcation is supercritical for ALL parameters.")

    return Re_c1


def numerical_verification():
    """Cross-validate symbolic formula against numerical computation."""
    import importlib.util

    Re_c1, (k, C, rho, a0, a1) = compute_Re_c1_symbolic()

    spec = importlib.util.spec_from_file_location(
        'hopf_lyapunov',
        'experiments/src/autonomy_lab/hopf_lyapunov.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("\n" + "=" * 72)
    print("  NUMERICAL CROSS-VALIDATION")
    print("=" * 72)

    print(f"\n  {'a':>5} {'C':>5} {'k':>5} {'xc':>5} {'rho':>5} | "
          f"{'sym':>12} {'num':>12} {'match':>6}")
    print("  " + "-" * 65)

    mismatches = 0
    total = 0

    for a_val in [0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 4.5]:
        for C_val in [2.0, 5.0, 10.0]:
            if a_val >= C_val:
                continue
            for k_val in [2.0, 5.0, 10.0, 20.0, 50.0]:
                for xc_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    rho_val = a_val / C_val
                    x_star = xc_val + np.log(a_val / (C_val - a_val)) / k_val
                    if not (0.001 < x_star < 0.999):
                        continue
                    a0_val = x_star * (1 - x_star)
                    a1_val = 1 - 2*x_star

                    sym_val = float(Re_c1.subs([
                        (C, C_val), (k, k_val), (rho, rho_val),
                        (a0, a0_val), (a1, a1_val)
                    ]))

                    try:
                        r = mod.compute_hopf_lyapunov(a_val, C_val, k_val, xc_val)
                        num_val = r.Re_c1
                    except (ValueError, ZeroDivisionError):
                        continue

                    total += 1
                    match = abs(sym_val - num_val) < 1e-4 * max(abs(num_val), 1e-10)
                    if not match:
                        mismatches += 1

                    if not match or total <= 10:
                        print(f"  {a_val:5.1f} {C_val:5.1f} {k_val:5.1f} "
                              f"{xc_val:5.1f} {rho_val:5.2f} | "
                              f"{sym_val:12.4f} {num_val:12.4f} "
                              f"{'OK' if match else 'FAIL'}")

    print(f"\n  Total: {total} cases, {mismatches} mismatches")
    if mismatches == 0:
        print("  ALL MATCH.")

    # Dense sign scan
    print("\n  Dense sign scan (22,000+ combinations)...")
    positive = 0
    scanned = 0
    for a_val in np.linspace(0.1, 9.9, 20):
        for C_val in [2.0, 5.0, 10.0, 20.0]:
            if a_val >= C_val:
                continue
            for k_val in np.logspace(-0.5, 2.5, 30):
                for xc_val in np.linspace(0.05, 0.95, 20):
                    try:
                        r = mod.compute_hopf_lyapunov(a_val, C_val, k_val, xc_val)
                        scanned += 1
                        if r.Re_c1 >= 0:
                            positive += 1
                    except (ValueError, ZeroDivisionError):
                        pass

    print(f"  Scanned: {scanned}, positive Re(c_1): {positive}")
    if positive == 0:
        print(f"  CONFIRMED: Re(c_1) < 0 for all {scanned} tested cases.")


if __name__ == "__main__":
    print("#" * 72)
    print("#  SYMBOLIC PROOF: UNIVERSAL SUPERCRITICALITY")
    print("#  Delayed replicator equation Hopf bifurcation")
    print("#" * 72)

    verify_bvp_symbolic()
    prove_sign()
    numerical_verification()

    print("\n" + "#" * 72)
    print("#  PROOF COMPLETE")
    print("#" * 72)
