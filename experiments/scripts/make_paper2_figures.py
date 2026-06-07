#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'paper2'
NOISE_DIR = RESULTS / 'noise_sweep_50'
TOPO_DIR = RESULTS / 'topology_bridge_50'
GOV_DIR = RESULTS / 'governance_loss_50'
FIGDIR = ROOT.parent / 'paper2' / 'latex' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)


def fig_noise_sweep():
    summary = pd.read_csv(NOISE_DIR / 'summary.csv')
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
    fig.suptitle('Paper 2: Noise Mode Comparison')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_noise_sweep.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_noise_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {FIGDIR / "fig_noise_sweep.pdf"}')


def fig_topology_bridge():
    summary = pd.read_csv(TOPO_DIR / 'summary.csv')
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
        axes[2].legend(['Bridge radical', 'Dangerous bridge radical', 'Punished'], fontsize=7)
    fig.suptitle('Paper 2: Bridge Targeting by Topology')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_topology_bridge.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_topology_bridge.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {FIGDIR / "fig_topology_bridge.pdf"}')


def fig_governance_loss():
    summary = pd.read_csv(GOV_DIR / 'summary.csv')
    agg = summary.groupby('noise_mode').agg(
        usefulness=('usefulness_mean_tail', 'mean'),
        punished=('punished_fraction_mean_tail', 'mean'),
        fp_bridge=('false_positive_bridge_rate_mean_tail', 'mean'),
        fn_bridge=('false_negative_bridge_rate_mean_tail', 'mean'),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(agg['noise_mode'], agg['usefulness'], color='steelblue')
    axes[0].set_ylabel('Mean Usefulness')
    axes[0].set_title('Usefulness by Noise Mode')

    x = np.arange(len(agg))
    w = 0.35
    axes[1].bar(x - w/2, agg['fp_bridge'], w, label='FP (bridge)', color='coral')
    axes[1].bar(x + w/2, agg['fn_bridge'], w, label='FN (bridge)', color='teal')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(agg['noise_mode'], rotation=15)
    axes[1].set_ylabel('Rate')
    axes[1].set_title('Bridge Error Rates')
    axes[1].legend()

    fig.suptitle('Paper 2: Governance Loss Decomposition')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_governance_loss.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_governance_loss.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {FIGDIR / "fig_governance_loss.pdf"}')


if __name__ == '__main__':
    print('Generating Paper 2 figures...')
    fig_noise_sweep()
    fig_topology_bridge()
    fig_governance_loss()
    print('Done.')
