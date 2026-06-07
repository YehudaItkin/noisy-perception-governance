#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autonomy_lab.simulation import run_simulation
from autonomy_lab.metrics import summarize_history
from _sweep_utils import load_config, expand_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, run_cfg in enumerate(expand_runs(cfg)):
        run_name = f"run_{idx:04d}_seed_{run_cfg.get('seed')}"
        rdir = outdir / run_name
        rdir.mkdir(parents=True, exist_ok=True)
        history = run_simulation(run_cfg)
        pd.DataFrame(history).to_csv(rdir / 'history.csv', index=False)
        summary = summarize_history(history)
        (rdir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (rdir / 'config_resolved.json').write_text(json.dumps(run_cfg, indent=2), encoding='utf-8')
        rows.append({**run_cfg, **summary})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'summary.csv', index=False)
    print(f"wrote {len(df)} runs to {outdir}")
if __name__ == '__main__': main()
