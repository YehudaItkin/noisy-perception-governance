#!/usr/bin/env python3
"""Experiment 10 robustness: p_out sweep for the bridge-contagion null.

The headline Exp 10 null (bridge placement of dangerous content does not raise
population spread) was obtained at p_out=0.004 -- a near-disconnected SBM. That
choice mechanically limits how much a bridge can amplify cross-community spread,
so the null could be a sparsity artifact rather than a property of bridges. This
script reruns the bridge- vs non-bridge-biased placement contrast across a range
of inter-community densities p_out, holding everything else fixed (FN-heavy noise,
dangerous fraction 0.20). If denser bridges revive the premise, the difference
should grow with p_out; if it stays ~0, the null is robust to sparsity.
"""
import sys, argparse, csv, statistics as st
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from autonomy_lab.simulation import run_simulation

BASE = {
    "steps": 500, "n": 240, "topology": "modular", "num_communities": 6,
    "p_in": 0.08, "bridge_fraction": 0.12,
    "delay_steps": 6, "k": 10.0, "alarm_threshold": 0.72, "influence_weight": 0.22,
    "learning": True, "regulator": "static", "regulator_force": 1.0,
    "enable_content": True, "enable_noise": True, "noise_mode": "fn_heavy",
    "bridge_multiplier": 1.0,
}

P_OUT_LEVELS = [0.004, 0.01, 0.02, 0.04]


def tail_mean(series, frac=0.5):
    n = len(series); burn = int(n * frac)
    return float(np.mean(series[burn:])) if n > burn else float('nan')


def welch_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, float('nan')
    t = (a.mean() - b.mean()) / se
    df = se**4 / ((va / na)**2 / (na - 1) + (vb / nb)**2 / (nb - 1))
    return float(t), float(df)


def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=50)
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p_out in P_OUT_LEVELS:
        for placement in ['bridge_biased', 'nonbridge_biased']:
            for seed in range(1, args.seeds + 1):
                cfg = dict(BASE)
                cfg.update({'p_out': p_out, 'content_placement': placement, 'seed': seed})
                h = run_simulation(cfg)
                rows.append({
                    'p_out': p_out, 'placement': placement, 'seed': seed,
                    'radical_fraction': tail_mean(h['R']),
                    'bridge_radical': tail_mean(h.get('bridge_radical_fraction', [0])),
                })

    with open(outdir / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"{'p_out':>7} | {'bridge R':>10} | {'nonbridge R':>12} | {'diff':>8} | {'t':>6} | {'d':>6}")
    print("-" * 64)
    for p_out in P_OUT_LEVELS:
        br = [r['radical_fraction'] for r in rows if r['p_out'] == p_out and r['placement'] == 'bridge_biased']
        nb = [r['radical_fraction'] for r in rows if r['p_out'] == p_out and r['placement'] == 'nonbridge_biased']
        t, _ = welch_t(br, nb)
        d = cohen_d(br, nb)
        print(f"{p_out:>7.3f} | {st.mean(br):>10.4f} | {st.mean(nb):>12.4f} | "
              f"{st.mean(br) - st.mean(nb):>+8.4f} | {t:>6.2f} | {d:>6.2f}")


if __name__ == '__main__':
    main()
