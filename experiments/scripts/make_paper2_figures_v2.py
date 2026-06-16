#!/usr/bin/env python3
"""Generate Paper 2 figures from v2 (500-step) data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'paper2_v2'
# Canonical, deposited result tree (matches the paper text and the verifier).
# All Paper-2 figures read from here so the deposit reproduces them exactly.
CANON = ROOT / 'results' / 'paper2'
FIGDIR = ROOT.parent / 'paper2' / 'latex' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)


def fig_noise_sweep():
    summary = pd.read_csv(CANON / 'noise_sweep_50' / 'summary.csv')
    metrics = ['false_positive_rate_mean_tail', 'false_negative_rate_mean_tail',
               'false_positive_bridge_rate_mean_tail', 'false_negative_bridge_rate_mean_tail',
               'usefulness_mean_tail', 'punished_fraction_mean_tail']
    labels = ['FP (all)', 'FN (all)', 'FP (bridge)', 'FN (bridge)', 'Usefulness', 'Punished frac']

    agg = summary.groupby('noise_mode')[metrics].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agg))
    width = 0.12
    for i, (col, lbl) in enumerate(zip(metrics, labels)):
        ax.bar(x + i * width, agg[col], width, label=lbl)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(agg.index, rotation=15)
    ax.set_ylabel('Rate / Value')
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle('Noise Mode Comparison (500 steps, 50 seeds)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_noise_sweep.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_noise_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved fig_noise_sweep')


def fig_topology_bridge():
    summary = pd.read_csv(CANON / 'topology_bridge_50' / 'summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for idx, topo in enumerate(['er', 'modular', 'scale_free']):
        sub = summary[summary['topology'] == topo]
        agg = sub.groupby('bridge_multiplier')[['bridge_radical_fraction_mean_tail',
                                                 'dangerous_radical_bridge_fraction_mean_tail',
                                                 'punished_fraction_mean_tail']].mean()
        agg.plot(kind='bar', ax=axes[idx], legend=(idx == 2))
        axes[idx].set_title(topo)
        axes[idx].set_xlabel('Bridge multiplier')
    axes[0].set_ylabel('Fraction')
    if axes[2].get_legend():
        axes[2].legend(['Bridge radical', 'Dangerous bridge', 'Punished'], fontsize=7)
    fig.suptitle('Bridge Targeting by Topology (500 steps, 50 seeds)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_topology_bridge.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_topology_bridge.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved fig_topology_bridge')


def fig_governance_loss():
    # Decomposition must show the SAME L_gov reported in Table 1 / Experiment 1,
    # i.e. the canonical noise sweep at the default targeting multiplier m=1.35,
    # not the multiplier grid. The multiplier-robustness check stays in the prose,
    # backed by governance_loss_50.
    summary = pd.read_csv(CANON / 'noise_sweep_50' / 'summary.csv')
    summary = summary[summary['bridge_multiplier'] == 1.35].copy()
    summary['L_fn'] = summary['false_negative_bridge_rate_mean_tail'] * summary['dangerous_radical_bridge_fraction_mean_tail']
    summary['L_fp'] = summary['false_positive_bridge_rate_mean_tail'] * summary['punished_fraction_mean_tail']
    summary['L_control'] = summary['avg_repression_mean_tail'] ** 2
    summary['L_gov'] = summary['L_fn'] + summary['L_fp'] + summary['L_control']

    modes = ['oracle', 'default', 'fp_heavy', 'fn_heavy']
    labels = ['Oracle', 'Default', 'FP-heavy', 'FN-heavy']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    means_fn = [summary[summary['noise_mode'] == m]['L_fn'].mean() for m in modes]
    means_fp = [summary[summary['noise_mode'] == m]['L_fp'].mean() for m in modes]
    means_ctrl = [summary[summary['noise_mode'] == m]['L_control'].mean() for m in modes]
    x = np.arange(len(modes))
    axes[0].bar(x, means_ctrl, label='$L_{control}$', color='steelblue')
    axes[0].bar(x, means_fp, bottom=means_ctrl, label='$L_{FP}$', color='coral')
    axes[0].bar(x, means_fn, bottom=[c + f for c, f in zip(means_ctrl, means_fp)], label='$L_{FN}$', color='teal')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylabel('Governance Loss')
    axes[0].set_title('Governance Loss Decomposition')
    axes[0].legend(fontsize=8)

    for i, (m, lbl) in enumerate(zip(modes, labels)):
        sub = summary[summary['noise_mode'] == m]
        axes[1].scatter(sub['usefulness_mean_tail'], sub['L_gov'], alpha=0.4, s=15, label=lbl)
    axes[1].set_xlabel('Usefulness')
    axes[1].set_ylabel('Governance Loss')
    axes[1].set_title('Usefulness vs Governance Loss')
    axes[1].legend(fontsize=8)

    data = [summary[summary['noise_mode'] == m]['L_gov'].values for m in modes]
    bp = axes[2].boxplot(data, tick_labels=labels, patch_artist=True)
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#1abc9c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[2].set_ylabel('Governance Loss')
    axes[2].set_title('$L_{gov}$ Distribution')
    axes[2].tick_params(axis='x', rotation=15)

    fig.suptitle('Governance Loss Reveals Hidden Cost of Classification Error (500 steps)', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_governance_loss_decomp.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_governance_loss_decomp.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved fig_governance_loss_decomp')


if __name__ == '__main__':
    print('Generating Paper 2 v2 figures...')
    fig_noise_sweep()
    fig_topology_bridge()
    fig_governance_loss()
    print('Done.')
