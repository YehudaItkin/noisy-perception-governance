"""
Center manifold reduction and first Lyapunov coefficient for the delayed
replicator equation:

    ẋ(t) = x(t)(1-x(t))[a - C·p(x(t-Δ))]

where p(x) = sigmoid(k(x - x_c)).

Uses the Hassard–Kazarinoff–Wan formalism adapted to scalar DDEs.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class HopfResult:
    x_star: float
    b: float
    omega: float
    tau_c: float
    g20: complex
    g11: complex
    g02: complex
    g21: complex
    c1: complex
    Re_c1: float
    direction: str
    mu2: float
    beta2: float
    T2: float


def sigmoid_derivatives(k: float, rho: float):
    """Derivatives of p(x) = sigmoid(k(x-x_c)) at x* where p(x*) = rho."""
    q = rho * (1 - rho)
    p1 = k * q
    p2 = k**2 * q * (1 - 2*rho)
    p3 = k**3 * q * (1 - 6*rho*(1-rho))
    return p1, p2, p3


def compute_hopf_lyapunov(a: float, C: float, k: float, x_c: float) -> HopfResult:
    """
    Compute the first Lyapunov coefficient c₁(0) for the Hopf bifurcation
    of the delayed replicator equation at the critical delay Δ_c.

    Returns HopfResult with direction ('supercritical' or 'subcritical').

    Theory
    ------
    The DDE linearized about x* is  ẋ(t) = β u(t-Δ)  with β = -b < 0,
    which is the Hayes equation.  At Δ_c = π/(2b) a conjugate pair ±ib
    crosses the imaginary axis.

    We expand the nonlinearity to third order, project onto the 2-D center
    eigenspace using the Hale–Verduyn Lunel bilinear form, solve for the
    center manifold corrections W₂₀, W₁₁, and assemble the normal form
    coefficient  c₁(0) = (i/2ω)(g₂₀g₁₁ - 2|g₁₁|² - |g₀₂|²/3) + g₂₁/2.

    Sign convention: Re(c₁) < 0  ⟹  supercritical (stable limit cycle).
    """
    rho = a / C

    # --- Interior equilibrium ---
    if rho <= 0 or rho >= 1:
        raise ValueError(f"No interior equilibrium: a/C = {rho}")
    x_star = x_c + np.log(a / (C - a)) / k
    if not 0 < x_star < 1:
        raise ValueError(f"x* = {x_star} outside (0,1)")

    alpha_0 = x_star * (1 - x_star)       # x*(1 - x*)
    alpha_1 = 1 - 2*x_star                # 1 - 2x*

    # --- Sigmoid derivatives at equilibrium ---
    p1, p2, p3 = sigmoid_derivatives(k, rho)

    # --- Linear coefficient and critical delay ---
    b = C * alpha_0 * p1
    omega = b
    tau_c = np.pi / (2 * b)

    # --- Adjoint eigenvector normalization ---
    # Bilinear form gives ⟨ψ, φ⟩ = D̄ · Δ'(iω) = 1
    # where Δ'(iω) = 1 + iπ/2  (derivative of char. eqn at bifurcation)
    D_bar = 1 / (1 + 1j*np.pi/2)
    D     = np.conj(D_bar)

    # =====================================================================
    #  SECOND-ORDER: quadratic nonlinearity f₂(u, u_d)
    # =====================================================================
    # f₂ = F_ab · u · u_d  +  ½ F_bb · u_d²
    # where F_ab = -C α₁ p₁,   F_bb = -C α₀ p₂
    #
    # On center manifold (leading order):
    #   u   = z + z̄
    #   u_d = -iz + iz̄ = i(z̄ - z)     [since e^{-iωτ_c} = -i]
    #
    # Expanding f₂ as  f₂₍₂₀₎ z² + f₂₍₁₁₎ zz̄ + f₂₍₀₂₎ z̄²:

    f20 =  1j*C*alpha_1*p1 + C*alpha_0*p2/2
    f11 = -C*alpha_0*p2
    f02 = -1j*C*alpha_1*p1 + C*alpha_0*p2/2

    g20 = 2 * D_bar * f20
    g11 =     D_bar * f11
    g02 = 2 * D_bar * f02
    g02_bar = np.conj(g02)

    # =====================================================================
    #  CENTER MANIFOLD: W₂₀(θ) and W₁₁(θ)
    # =====================================================================
    #
    # W₂₀ satisfies (2iω - A)W₂₀ = H₂₀  with:
    #   interior ODE solution (θ < 0):
    #     W₂₀(θ) = (ig₂₀/ω)e^{iωθ} + (iḡ₀₂/(3ω))e^{-iωθ} + E₁ e^{2iωθ}
    #   boundary at θ = 0 determines E₁.
    #
    # Boundary condition:
    #   2iω W₂₀(0) + b W₂₀(-τ) = 2f₂₀ - g₂₀ - ḡ₀₂
    # ⟹  E₁ = 2f₂₀ / [b(2i - 1)]

    E1 = 2*f20 / (b * (2j - 1))

    W20_0   = 1j*g20/omega + 1j*g02_bar/(3*omega) + E1
    W20_tau = g20/omega - g02_bar/(3*omega) - E1

    # W₁₁ satisfies  -A W₁₁ = H₁₁:
    #   W₁₁(θ) = (g₁₁/(iω))e^{iωθ} - (ḡ₁₁/(iω))e^{-iωθ} + E₂
    #   E₂ = f₁₁ / b

    g11_bar = np.conj(g11)
    E2 = f11 / b

    W11_0   = (g11 - g11_bar) / (1j*omega) + E2
    W11_tau = -(g11 + g11_bar) / omega + E2

    # =====================================================================
    #  THIRD-ORDER: cubic nonlinearity + W corrections → g₂₁
    # =====================================================================
    #
    # f₃(u, u_d) = C p₁ u² u_d  -  ½ C α₁ p₂ u u_d²  -  ⅙ C α₀ p₃ u_d³
    #
    # z²z̄ coefficient on leading-order center manifold:
    f3_21 = -1j*C*p1 - 0.5*C*alpha_1*p2 + 0.5j*C*alpha_0*p3

    # W corrections from f₂ evaluated with center manifold corrections:
    #
    # From  -C α₁ p₁ · u · u_d  differentiated:
    #   piece1 = -iCα₁p₁ · [W₂₀(0)/2 - W₁₁(0)]          (u_corr × u_d_lead)
    #   piece2 = -Cα₁p₁  · [W₂₀(-τ)/2 + W₁₁(-τ)]        (u_lead × u_d_corr)
    #
    # From  -½ C α₀ p₂ · u_d²  differentiated:
    #   piece3 = -iCα₀p₂ · [W₂₀(-τ)/2 - W₁₁(-τ)]        (u_d_corr × u_d_lead)

    piece1 = -1j * C * alpha_1 * p1 * (W20_0/2 - W11_0)
    piece2 = -C * alpha_1 * p1 * (W20_tau/2 + W11_tau)
    piece3 = -1j * C * alpha_0 * p2 * (W20_tau/2 - W11_tau)

    C_W = piece1 + piece2 + piece3

    f_21 = f3_21 + C_W
    g21 = 2 * D_bar * f_21

    # =====================================================================
    #  FIRST LYAPUNOV COEFFICIENT
    # =====================================================================
    c1 = (1j / (2*omega)) * (g20*g11 - 2*abs(g11)**2 - abs(g02)**2/3) + g21/2

    Re_c1 = c1.real

    # --- Bifurcation direction and stability ---
    # Transversality: Re(dλ/dΔ) = b²/(1 + π²/4) > 0
    alpha_prime = b**2 / (1 + np.pi**2/4)

    # μ₂ = -Re(c₁)/α'(0):  μ₂ > 0 ⟹ supercritical
    mu2 = -Re_c1 / alpha_prime

    # β₂ = 2 Re(c₁):  β₂ < 0 ⟹ stable periodic orbits
    beta2 = 2 * Re_c1

    # T₂ = -Im(c₁)/(ω·α'(0)):  period correction
    T2 = -c1.imag / (omega * alpha_prime)

    direction = "supercritical" if Re_c1 < 0 else "subcritical"

    return HopfResult(
        x_star=x_star, b=b, omega=omega, tau_c=tau_c,
        g20=g20, g11=g11, g02=g02, g21=g21,
        c1=c1, Re_c1=Re_c1, direction=direction,
        mu2=mu2, beta2=beta2, T2=T2,
    )


def _verify_boundary_conditions(a, C, k, x_c):
    """Verify that W₂₀ and W₁₁ satisfy their boundary conditions."""
    rho = a / C
    x_star = x_c + np.log(a / (C - a)) / k
    alpha_0 = x_star * (1 - x_star)
    alpha_1 = 1 - 2*x_star
    p1, p2, p3 = sigmoid_derivatives(k, rho)
    b = C * alpha_0 * p1
    omega = b
    D_bar = 1 / (1 + 1j*np.pi/2)
    D = np.conj(D_bar)

    f20 = 1j*C*alpha_1*p1 + C*alpha_0*p2/2
    f11 = -C*alpha_0*p2
    g20 = 2 * D_bar * f20
    g11 = D_bar * f11
    g02 = 2 * D_bar * (-1j*C*alpha_1*p1 + C*alpha_0*p2/2)
    g02_bar = np.conj(g02)

    E1 = 2*f20 / (b * (2j - 1))
    W20_0   = 1j*g20/omega + 1j*g02_bar/(3*omega) + E1
    W20_tau = g20/omega - g02_bar/(3*omega) - E1

    lhs_20 = 2j*omega*W20_0 + b*W20_tau
    rhs_20 = 2*f20 - g20 - g02_bar
    err_20 = abs(lhs_20 - rhs_20)

    g11_bar = np.conj(g11)
    E2 = f11 / b
    W11_tau = -(g11 + g11_bar) / omega + E2

    lhs_11 = b * W11_tau
    rhs_11 = f11 * (1 - D_bar - D)
    err_11 = abs(lhs_11 - rhs_11)

    return err_20, err_11


if __name__ == "__main__":
    print("=" * 72)
    print("  CENTER MANIFOLD REDUCTION FOR DELAYED REPLICATOR EQUATION")
    print("  First Lyapunov coefficient at the Hopf bifurcation")
    print("=" * 72)

    # --- Default parameters from Paper 1 ---
    params = dict(a=2.0, C=5.0, k=10.0, x_c=0.5)
    print(f"\nDefault parameters: a={params['a']}, C={params['C']}, "
          f"k={params['k']}, x_c={params['x_c']}")

    r = compute_hopf_lyapunov(**params)
    err20, err11 = _verify_boundary_conditions(**params)

    print(f"\n  x*     = {r.x_star:.6f}")
    print(f"  b = ω  = {r.b:.6f}")
    print(f"  Δ_c    = {r.tau_c:.6f}")

    print(f"\n  g₂₀    = {r.g20:.6f}")
    print(f"  g₁₁    = {r.g11:.6f}")
    print(f"  g₀₂    = {r.g02:.6f}")
    print(f"  g₂₁    = {r.g21:.6f}")

    print(f"\n  c₁(0)  = {r.c1:.6f}")
    print(f"  Re(c₁) = {r.Re_c1:.6f}")
    print(f"  μ₂     = {r.mu2:.6f}  (>0 ⟹ supercritical)")
    print(f"  β₂     = {r.beta2:.6f}  (<0 ⟹ stable orbits)")
    print(f"  T₂     = {r.T2:.6f}  (period correction)")

    print(f"\n  ⟹ Direction: {r.direction.upper()}")

    print(f"\n  Boundary condition errors:  W₂₀: {err20:.2e},  W₁₁: {err11:.2e}")

    # --- Parameter sweep ---
    print("\n" + "=" * 72)
    print("  PARAMETER SWEEP")
    print("=" * 72)

    print(f"\n{'a':>6} {'C':>6} {'k':>6} {'x_c':>6} │ {'x*':>8} {'Δ_c':>8} {'Re(c₁)':>12} │ {'Direction':>14}")
    print("─" * 72)

    sweep_params = [
        (2.0, 5.0, 10.0, 0.5),
        (2.0, 5.0, 5.0,  0.5),
        (2.0, 5.0, 20.0, 0.5),
        (2.0, 5.0, 50.0, 0.5),
        (1.0, 5.0, 10.0, 0.5),
        (3.0, 5.0, 10.0, 0.5),
        (4.0, 5.0, 10.0, 0.5),
        (2.5, 5.0, 10.0, 0.5),  # rho = 0.5 (symmetric)
        (2.0, 5.0, 10.0, 0.3),
        (2.0, 5.0, 10.0, 0.7),
        (1.0, 3.0, 10.0, 0.5),
        (2.0, 3.0, 10.0, 0.5),
        (1.0, 10.0, 10.0, 0.5),
        (5.0, 10.0, 10.0, 0.5),  # rho = 0.5
        (1.0, 2.0, 10.0, 0.5),
        (0.5, 5.0, 10.0, 0.5),
        (4.5, 5.0, 10.0, 0.5),
    ]

    for a, C, k, x_c in sweep_params:
        try:
            r = compute_hopf_lyapunov(a, C, k, x_c)
            print(f"{a:6.1f} {C:6.1f} {k:6.1f} {x_c:6.1f} │ {r.x_star:8.4f} "
                  f"{r.tau_c:8.4f} {r.Re_c1:12.6f} │ {r.direction:>14}")
        except ValueError as e:
            print(f"{a:6.1f} {C:6.1f} {k:6.1f} {x_c:6.1f} │ {'N/A':>8} "
                  f"{'N/A':>8} {'N/A':>12} │ {str(e):>14}")

    # --- Special case: symmetric (rho = 1/2) analysis ---
    print("\n" + "=" * 72)
    print("  SYMMETRIC CASE: ρ = a/C = 1/2  (p₂ = 0, g₁₁ = 0)")
    print("=" * 72)

    r_sym = compute_hopf_lyapunov(2.5, 5.0, 10.0, 0.5)
    print(f"\n  When ρ = 1/2:")
    print(f"    p₂ = k²ρ(1-ρ)(1-2ρ) = 0 exactly")
    print(f"    g₁₁ = {r_sym.g11:.6f}  (should be ≈ 0)")
    print(f"    f₁₁ = -Cα₀p₂ = 0")
    print(f"    c₁ simplifies to: -i|g₀₂|²/(6ω) + g₂₁/2")
    print(f"    Re(c₁) = {r_sym.Re_c1:.6f}  → {r_sym.direction}")
