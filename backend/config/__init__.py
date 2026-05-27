import yaml
from pathlib import Path

_CONFIG = None

def load_config(path: str = None) -> dict:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    if path is None:
        path = Path(__file__).parent / "params.yaml"
    with open(path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG

def reset_config():
    global _CONFIG
    _CONFIG = None

def get(section: str, key: str):
    cfg = load_config()
    return cfg[section][key]