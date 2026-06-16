# Noisy-Perception Governance

Code and data for **"Selective Control under Noisy Perception: Governance Failures Hidden by
Aggregate Metrics in Modular Networks"** (the *governance loss* paper). A lightweight agent-based simulation in which a regulator governs a
networked population through an imperfect classifier, plus the analysis that produces every figure and
table in the paper.

> **Status:** public. Preprint: [arXiv:2606.14819](https://arxiv.org/abs/2606.14819).
> Companion paper (institutional delay without classification noise):
> [`delayed-repression-instability`](https://github.com/YehudaItkin/delayed-repression-instability).

## What this is

A regulator cannot observe the true type of each agent's content and must act on a noisy classifier
label. Errors are filtered through network position: a small set of **bridge** nodes link otherwise
separated communities, so misclassifications there cost far more than misclassifications in a
community's interior. We introduce a **governance loss** decomposition
`L_gov = L_FN + L_FP + L_control` that is sensitive to *where* errors fall, and show that aggregate
performance metrics are blind to failures this loss detects.

## Install

```bash
pip install -r requirements.txt          # numpy, pandas, networkx, matplotlib
```

All commands below assume the working directory is the repository root.

## Reproduce the results

Each experiment writes a per-run `history.csv`, a `summary.json`, a `config_resolved.json`, and an
aggregated `summary.csv`. The raw per-run outputs are large and regenerable, so **only the
aggregated `summary.csv` files are tracked** (under `experiments/results/paper2/<name>/`). Re-running a
sweep regenerates the full output locally.

```bash
# smoke test
python experiments/scripts/run_single.py --config experiments/configs/default.json --out experiments/results/smoke

# the paper's sweeps (results land in experiments/results/paper2/<name>/)
python experiments/scripts/run_paper2_noise_sweep.py        --config experiments/configs/paper2_noise_sweep.json        --out experiments/results/paper2/noise_sweep_50
python experiments/scripts/run_paper2_topology_bridge.py    --config experiments/configs/paper2_topology_bridge.json    --out experiments/results/paper2/topology_bridge_50
python experiments/scripts/run_paper2_governance_loss.py    --config experiments/configs/paper2_governance_loss.json    --out experiments/results/paper2/governance_loss_50
python experiments/scripts/run_paper2_adaptive_multiplier.py --config experiments/configs/paper2_adaptive_multiplier.json --out experiments/results/paper2/adaptive_multiplier
python experiments/scripts/run_paper2_delay_x_noise.py      --config experiments/configs/paper2_delay_x_noise.json      --out experiments/results/paper2/delay_x_noise
python experiments/scripts/run_paper2_endogenous_content.py --config experiments/configs/paper2_endogenous_content.json --out experiments/results/paper2/endogenous_content
python experiments/scripts/run_paper2_coupling_sweep.py     --config experiments/configs/paper2_coupling_sweep.json     --out experiments/results/paper2/coupling_sweep
python experiments/scripts/run_paper2_adaptive_delay.py     --config experiments/configs/paper2_adaptive_delay.json     --out experiments/results/paper2/adaptive_delay
python experiments/scripts/run_paper2_joint_optimization.py --config experiments/configs/paper2_joint_optimization.json --out experiments/results/paper2/joint_optimization
python experiments/scripts/run_paper2_bridge_contagion.py   --out experiments/results/paper2/bridge_contagion   # built-in config (--seeds optional)

# figures (read the tracked summary.csv files)
python experiments/scripts/make_paper2_figures.py
python experiments/scripts/make_paper2_figures_v2.py
python experiments/scripts/make_paper2_new_figures.py
```

## Experiment ↔ paper map

| Paper | Sweep directory | Script |
|-------|-----------------|--------|
| Exp. 1 (noise sweep) | `noise_sweep_50` | `run_paper2_noise_sweep.py` |
| Exp. 2 (topology × targeting) | `topology_bridge_50` | `run_paper2_topology_bridge.py` |
| Exp. 3 (loss decomposition) | `governance_loss_50` | `run_paper2_governance_loss.py` |
| Exp. 4 (adaptive vs static) | `adaptive_multiplier` | `run_paper2_adaptive_multiplier.py` |
| Exp. 5 (delay × noise) | `delay_x_noise` | `run_paper2_delay_x_noise.py` |
| Exp. 6 (endogenous content) | `endogenous_content` | `run_paper2_endogenous_content.py` |
| Exp. 7 (coupling threshold) | `coupling_sweep` | `run_paper2_coupling_sweep.py` |
| Exp. 8 (adaptive under delay) | `adaptive_delay` | `run_paper2_adaptive_delay.py` |
| Exp. 9 (joint optimization) | `joint_optimization` | `run_paper2_joint_optimization.py` |
| Exp. 10 (bridge contagion) | `bridge_contagion` | `run_paper2_bridge_contagion.py` |
| Discussion robustness | `bridge_fraction_sweep`, `lambda_infl_sweep` | (see scripts) |

## Layout

```
experiments/
├── src/autonomy_lab/     # simulation library (graph, agents, step, metrics, theory, io)
├── scripts/              # run_paper2_*.py sweeps + make_paper2_*.py figure builders
├── configs/              # JSON experiment configs (default.json + paper2_*.json)
└── results/paper2/       # aggregated summary.csv per experiment (raw runs gitignored)
figures/                  # compiled paper figures (PDF)
```

## Citation

If you use this code, please cite the preprint:

> Igor Itkin. *Selective Control under Noisy Perception: Governance Failures Hidden by Aggregate Metrics in Modular Networks.* arXiv:2606.14819, 2026. <https://arxiv.org/abs/2606.14819>

```bibtex
@misc{itkin2026selective,
  title         = {Selective Control under Noisy Perception: Governance Failures Hidden by Aggregate Metrics in Modular Networks},
  author        = {Itkin, Igor},
  year          = {2026},
  eprint        = {2606.14819},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA}
}
```

Released under the MIT License (see [`LICENSE`](LICENSE)).
