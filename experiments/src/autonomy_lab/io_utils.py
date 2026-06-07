from __future__ import annotations
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding='utf-8')
