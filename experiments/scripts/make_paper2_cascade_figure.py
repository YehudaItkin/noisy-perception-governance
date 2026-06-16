#!/usr/bin/env python3
"""Figure for Experiment 11 (cascade consequence): peak dangerous-content reach by
placement across the simple->complex threshold, plus the betweenness-vs-degree
effect-size comparison."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'paper2' / 'cascade_consequence'
FIGDIR = ROOT.parent / 'paper2' / 'latex' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

LABELS = {'bridge_biased': 'Bridge (high betweenness)',
          'highdeg_nonbridge': 'High-degree non-bridge',
          'nonbridge_biased': 'Random non-bridge'}
COLORS = {'bridge_biased': 'C3', 'highdeg_nonbridge': 'C0', 'nonbridge_biased': 'C7'}


def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def main():
    df = pd.read_csv(RESULTS / 'summary.csv')
    thetas = sorted(df['theta'].unique())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: peak reach by placement vs theta
    for pl in ('bridge_biased', 'highdeg_nonbridge', 'nonbridge_biased'):
        means, ses = [], []
        for th in thetas:
            v = df[(df['theta'] == th) & (df['placement'] == pl)]['peak_D'].values
            means.append(v.mean()); ses.append(v.std(ddof=1) / np.sqrt(len(v)))
        axL.errorbar(thetas, means, yerr=ses, marker='o', capsize=3,
                     color=COLORS[pl], label=LABELS[pl])
    axL.set_xlabel(r'Contagion threshold $\theta$ (simple $\rightarrow$ complex)')
    axL.set_ylabel('Peak dangerous-content fraction')
    axL.legend(fontsize=8)
    axL.set_title('(a) Dangerous-content reach by seed placement')

    # Right: effect sizes vs theta
    dbn, dbh = [], []
    for th in thetas:
        bri = df[(df['theta'] == th) & (df['placement'] == 'bridge_biased')]['peak_D'].values
        nb = df[(df['theta'] == th) & (df['placement'] == 'nonbridge_biased')]['peak_D'].values
        hd = df[(df['theta'] == th) & (df['placement'] == 'highdeg_nonbridge')]['peak_D'].values
        dbn.append(cohen_d(bri, nb)); dbh.append(cohen_d(bri, hd))
    axR.plot(thetas, dbn, marker='s', color='C3', label='Bridge vs random non-bridge')
    axR.plot(thetas, dbh, marker='^', color='C0', label='Bridge vs high-degree non-bridge')
    axR.axhline(0, color='k', lw=0.6)
    axR.axhline(0.8, color='gray', ls=':', lw=0.8)
    axR.set_xlabel(r'Contagion threshold $\theta$')
    axR.set_ylabel("Cohen's $d$")
    axR.legend(fontsize=8)
    axR.set_title('(b) Effect of position: betweenness vs degree')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / f'fig_cascade_consequence.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved fig_cascade_consequence to {FIGDIR}')


if __name__ == '__main__':
    main()
