#!/usr/bin/env python3
"""Generate figures for Paper 2 Experiments 4-6 and temporal dynamics."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def governance_loss(row, lam_fn=1, lam_fp=1, lam_u=1):
    bfn = row.get('false_negative_bridge_rate_mean_tail', 0)
    bfp = row.get('false_positive_bridge_rate_mean_tail', 0)
    drb = row.get('dangerous_radical_bridge_fraction_mean_tail', 0)
    pun = row.get('punished_fraction_mean_tail', 0)
    avg_rep = row.get('avg_repression_mean_tail', 0)
    l_fn = lam_fn * bfn * drb
    l_fp = lam_fp * bfp * pun
    l_ctrl = lam_u * avg_rep ** 2
    return l_fn + l_fp + l_ctrl, l_fn, l_fp, l_ctrl


def load_summary(results_dir):
    p = Path(results_dir)
    csv = p / 'summary.csv'
    if csv.exists():
        return pd.read_csv(csv)
    rows = []
    for rdir in sorted(p.iterdir()):
        cfg_f = rdir / 'config_resolved.json'
        sum_f = rdir / 'summary.json'
        if cfg_f.exists() and sum_f.exists():
            cfg = json.loads(cfg_f.read_text())
            sm = json.loads(sum_f.read_text())
            rows.append({**cfg, **sm})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fig_adaptive_multiplier(results_dir, outdir):
    df = load_summary(results_dir)
    if df.empty:
        print("No adaptive multiplier data"); return

    df['L_gov'] = df.apply(lambda r: governance_loss(r)[0], axis=1)

    noise_modes = ['oracle', 'default', 'fp_heavy', 'fn_heavy']
    conditions = df['name'].unique() if 'name' in df.columns else []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Governance loss by condition x noise
    ax = axes[0]
    x = np.arange(len(noise_modes))
    width = 0.18
    for j, cond in enumerate(sorted(conditions)):
        means = []
        sems = []
        for nm in noise_modes:
            sub = df[(df['name'] == cond) & (df['noise_mode'] == nm)]
            means.append(sub['L_gov'].mean())
            sems.append(sub['L_gov'].std() / np.sqrt(max(len(sub), 1)))
        ax.bar(x + j * width, means, width, yerr=sems, label=cond, capsize=3)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(noise_modes)
    ax.set_ylabel('Governance Loss $\\mathcal{L}_{gov}$')
    ax.legend(fontsize=8)
    ax.set_title('Governance loss by enforcement policy')

    # Panel 2: Mean multiplier used by noise mode
    ax = axes[1]
    if 'bridge_multiplier_used_mean_tail' in df.columns:
        for cond in sorted(conditions):
            means = []
            for nm in noise_modes:
                sub = df[(df['name'] == cond) & (df['noise_mode'] == nm)]
                means.append(sub['bridge_multiplier_used_mean_tail'].mean())
            ax.plot(noise_modes, means, 'o-', label=cond)
        ax.set_ylabel('Mean bridge multiplier $m$')
        ax.legend(fontsize=8)
        ax.set_title('Bridge targeting intensity')

    plt.tight_layout()
    plt.savefig(outdir / 'fig_adaptive_multiplier.pdf')
    plt.close()
    print(f"Saved {outdir / 'fig_adaptive_multiplier.pdf'}")


def fig_delay_noise_heatmap(results_dir, outdir):
    df = load_summary(results_dir)
    if df.empty:
        print("No delay×noise data"); return

    df['L_gov'] = df.apply(lambda r: governance_loss(r)[0], axis=1)

    delays = sorted(df['delay_steps'].unique())
    noise_modes = ['oracle', 'default', 'fp_heavy', 'fn_heavy']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap 1: Governance loss
    grid = np.zeros((len(noise_modes), len(delays)))
    for i, nm in enumerate(noise_modes):
        for j, d in enumerate(delays):
            sub = df[(df['noise_mode'] == nm) & (df['delay_steps'] == d)]
            grid[i, j] = sub['L_gov'].mean()
    ax = axes[0]
    im = ax.imshow(grid, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels(delays)
    ax.set_yticks(range(len(noise_modes)))
    ax.set_yticklabels(noise_modes)
    ax.set_xlabel('Delay steps')
    ax.set_title('$\\mathcal{L}_{gov}$')
    plt.colorbar(im, ax=ax)

    # Heatmap 2: Runaway rate
    grid2 = np.zeros((len(noise_modes), len(delays)))
    for i, nm in enumerate(noise_modes):
        for j, d in enumerate(delays):
            sub = df[(df['noise_mode'] == nm) & (df['delay_steps'] == d)]
            grid2[i, j] = (sub['regime'] == 'runaway').mean() if len(sub) else 0
    ax = axes[1]
    im2 = ax.imshow(grid2, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(delays)))
    ax.set_xticklabels(delays)
    ax.set_yticks(range(len(noise_modes)))
    ax.set_yticklabels(noise_modes)
    ax.set_xlabel('Delay steps')
    ax.set_title('Runaway rate')
    plt.colorbar(im2, ax=ax)

    plt.tight_layout()
    plt.savefig(outdir / 'fig_delay_noise_heatmap.pdf')
    plt.close()
    print(f"Saved {outdir / 'fig_delay_noise_heatmap.pdf'}")


def fig_endogenous_content(results_dir, outdir):
    df = load_summary(results_dir)
    if df.empty:
        print("No endogenous content data"); return

    df['L_gov'] = df.apply(lambda r: governance_loss(r)[0], axis=1)

    noise_modes = ['oracle', 'default', 'fp_heavy']
    conditions = sorted(df['name'].unique()) if 'name' in df.columns else []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Content D fraction by condition x noise
    ax = axes[0]
    x = np.arange(len(noise_modes))
    width = 0.25
    for j, cond in enumerate(conditions):
        means = []
        for nm in noise_modes:
            sub = df[(df['name'] == cond) & (df['noise_mode'] == nm)]
            val = sub['content_frac_D_mean_tail'].mean() if 'content_frac_D_mean_tail' in sub.columns else 0
            means.append(val)
        ax.bar(x + j * width, means, width, label=cond)
    ax.set_xticks(x + width)
    ax.set_xticklabels(noise_modes)
    ax.set_ylabel('Dangerous content fraction (tail)')
    ax.axhline(y=0.20, color='gray', linestyle='--', linewidth=0.8, label='Initial (0.20)')
    ax.legend(fontsize=8)
    ax.set_title('Content type evolution')

    # Panel 2: Governance loss
    ax = axes[1]
    x = np.arange(len(noise_modes))
    for j, cond in enumerate(conditions):
        means = []
        sems = []
        for nm in noise_modes:
            sub = df[(df['name'] == cond) & (df['noise_mode'] == nm)]
            means.append(sub['L_gov'].mean())
            sems.append(sub['L_gov'].std() / np.sqrt(max(len(sub), 1)))
        ax.bar(x + j * width, means, width, yerr=sems, label=cond, capsize=3)
    ax.set_xticks(x + width)
    ax.set_xticklabels(noise_modes)
    ax.set_ylabel('Governance Loss $\\mathcal{L}_{gov}$')
    ax.legend(fontsize=8)
    ax.set_title('Governance loss with endogenous content')

    plt.tight_layout()
    plt.savefig(outdir / 'fig_endogenous_content.pdf')
    plt.close()
    print(f"Saved {outdir / 'fig_endogenous_content.pdf'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', default=str(ROOT / 'results' / 'paper2'))
    ap.add_argument('--outdir', default=str(ROOT.parent / 'paper2' / 'latex' / 'figures'))
    args = ap.parse_args()
    results_root = Path(args.results_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    am_dir = results_root / 'adaptive_multiplier'
    dn_dir = results_root / 'delay_x_noise'
    ec_dir = results_root / 'endogenous_content'

    if am_dir.exists():
        fig_adaptive_multiplier(am_dir, outdir)
    if dn_dir.exists():
        fig_delay_noise_heatmap(dn_dir, outdir)
    if ec_dir.exists():
        fig_endogenous_content(ec_dir, outdir)

    print("Done.")

if __name__ == '__main__':
    main()
