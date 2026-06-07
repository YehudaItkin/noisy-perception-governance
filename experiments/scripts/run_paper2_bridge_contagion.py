#!/usr/bin/env python3
"""Experiment 10: bridge contagion test (closes circularity concern L1).

Tests whether dangerous content placed on BRIDGE nodes produces more
population-level radicalization than the same content on NON-BRIDGE nodes,
holding the dangerous fraction fixed at 0.20. Run under FN-heavy noise so
that dangerous content is systematically missed (not punished) and free to
spread influence. If bridge placement yields higher radical fraction, bridge
errors have outsized DYNAMIC consequences -- independent of how L_gov weights
them by construction.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from autonomy_lab.simulation import run_simulation

BASE = {
    "steps": 500, "n": 240, "topology": "modular", "num_communities": 6,
    "p_in": 0.08, "p_out": 0.004, "bridge_fraction": 0.12,
    "delay_steps": 6, "k": 10.0, "alarm_threshold": 0.72, "influence_weight": 0.22,
    "learning": True, "regulator": "static", "regulator_force": 1.0,
    "enable_content": True, "enable_noise": True, "noise_mode": "fn_heavy",
    "bridge_multiplier": 1.0,
}

def tail_mean(series, frac=0.5):
    n = len(series); burn = int(n * frac)
    return float(np.mean(series[burn:])) if n > burn else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=50)
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for placement in ['bridge_biased', 'nonbridge_biased', 'uniform']:
        for seed in range(1, args.seeds + 1):
            cfg = dict(BASE); cfg.update({'content_placement': placement, 'seed': seed})
            h = run_simulation(cfg)
            rows.append({
                'placement': placement, 'seed': seed,
                'radical_fraction': tail_mean(h['R']),
                'bridge_radical': tail_mean(h.get('bridge_radical_fraction', [0])),
                'dangerous_radical_bridge': tail_mean(h.get('dangerous_radical_bridge_fraction', [0])),
            })
    import csv
    with open(outdir / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # Quick report
    import statistics as st
    print("placement          R_frac (mean ± sd)")
    for p in ['bridge_biased', 'nonbridge_biased', 'uniform']:
        vals = [r['radical_fraction'] for r in rows if r['placement'] == p]
        print(f"  {p:18s} {st.mean(vals):.4f} ± {st.pstdev(vals):.4f}  n={len(vals)}")

if __name__ == '__main__':
    main()
