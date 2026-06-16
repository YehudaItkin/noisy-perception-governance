#!/usr/bin/env python3
"""Experiment 11: does structural position carry outsized dynamic consequence?

Experiment 10 found that, with only one-hop influence, dangerous content on bridges
spreads no further than elsewhere -- so the bridge weighting in the governance loss
was an imported premise, not a model property. A one-hop reward cannot encode a
multi-hop, global property like betweenness. This experiment adds the missing
ingredient: a multi-hop dangerous-content contagion (apply_cascade), and asks
whether structural position now matters, across the simple->complex threshold axis
(cascade_threshold theta; Centola & Macy 2007) and against a degree-matched control
that separates betweenness from degree.

Placements (dangerous fraction fixed at 0.20):
  bridge_biased      -- D on highest-betweenness nodes (the metric's bridges)
  nonbridge_biased   -- D on random non-bridge nodes
  highdeg_nonbridge  -- D on highest-DEGREE non-bridge nodes (degree-matched control)

If bridge >> nonbridge but bridge ~ highdeg_nonbridge, the operative axis is degree,
not betweenness (the two are near-collinear in modular networks).
"""
import sys, argparse, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from autonomy_lab.simulation import run_simulation

BASE = {
    "steps": 500, "n": 240, "topology": "modular", "num_communities": 6,
    "p_in": 0.08, "p_out": 0.03, "bridge_fraction": 0.12,
    "delay_steps": 6, "k": 10.0, "alarm_threshold": 0.72, "influence_weight": 0.22,
    "learning": True, "regulator": "static", "regulator_force": 1.0,
    "enable_content": True, "enable_noise": True, "noise_mode": "fn_heavy",
    "bridge_multiplier": 1.0, "enable_cascade": True, "cascade_beta": 0.20,
}

PLACEMENTS = ['bridge_biased', 'nonbridge_biased', 'highdeg_nonbridge']
THETAS = [0.0, 0.2, 0.35, 0.5, 0.65]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=50)
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for theta in THETAS:
        for placement in PLACEMENTS:
            for seed in range(1, args.seeds + 1):
                cfg = dict(BASE)
                cfg.update({'cascade_threshold': theta, 'content_placement': placement, 'seed': seed})
                h = run_simulation(cfg)
                d = np.asarray(h['content_frac_D'], float)
                rows.append({
                    'theta': theta, 'placement': placement, 'seed': seed,
                    'peak_D': float(d.max()),
                    'early_D': float(d[:100].mean()),
                    'tail_D': float(d[len(d)//2:].mean()),
                    'peak_dcomm': float(np.max(h['d_communities'])),
                    'bridge_radical_tail': float(np.mean(h['bridge_radical_fraction'][len(d)//2:])),
                })

    with open(outdir / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def arr(theta, placement, col):
        return np.array([r[col] for r in rows if r['theta'] == theta and r['placement'] == placement])

    def cohen_d(a, b):
        sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
        return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0

    print(f"{'theta':>6} | {'bri peakD':>9} {'nb peakD':>9} {'hd peakD':>9} | {'d_bn':>6} {'d_bh':>6}")
    print("-" * 60)
    for theta in THETAS:
        bri, nb, hd = arr(theta, 'bridge_biased', 'peak_D'), arr(theta, 'nonbridge_biased', 'peak_D'), arr(theta, 'highdeg_nonbridge', 'peak_D')
        print(f"{theta:>6.2f} | {bri.mean():>9.4f} {nb.mean():>9.4f} {hd.mean():>9.4f} | "
              f"{cohen_d(bri,nb):>6.2f} {cohen_d(bri,hd):>6.2f}")
    print(f"\nWrote {len(rows)} rows to {outdir/'summary.csv'}")


if __name__ == '__main__':
    main()
