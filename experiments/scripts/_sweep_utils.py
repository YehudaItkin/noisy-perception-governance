from __future__ import annotations
import itertools, json
from pathlib import Path


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    base = {}
    if 'base' in cfg:
        base_path = Path(path).parent / cfg['base']
        base = json.loads(base_path.read_text())
    merged = {**base, **{k:v for k,v in cfg.items() if k != 'base'}}
    return merged


def expand_runs(cfg):
    seeds = cfg.get('seeds', [cfg.get('seed',1)])
    base = {k:v for k,v in cfg.items() if k not in ('sweep','conditions','seeds')}
    conditions = cfg.get('conditions') or [{'name':'default'}]
    sweep = cfg.get('sweep') or {}
    if sweep:
        keys = list(sweep.keys())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(sweep[k] for k in keys))]
    else:
        combos = [{}]
    runs = []
    for cond in conditions:
        for combo in combos:
            for seed in seeds:
                run = {**base, **cond, **combo, 'seed': seed}
                run['condition'] = cond.get('name', 'default')
                runs.append(run)
    return runs
