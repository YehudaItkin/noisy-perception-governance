#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, itertools
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from autonomy_lab.simulation import run_simulation
from autonomy_lab.metrics import summarize_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    history = run_simulation(cfg)
    pd.DataFrame(history).to_csv(outdir / 'history.csv', index=False)
    summary = summarize_history(history)
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (outdir / 'config_resolved.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__': main()
