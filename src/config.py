"""Konfigurasi eksperimen: load dari YAML/JSON dan bantu membenamkan dict ke notebook.

Sumber kebenaran konfigurasi = file di `configs/`. Generator membaca file ini,
lalu membenamkan representasi literal dict-nya ke sel Config notebook (sel 6),
sehingga notebook Kaggle tidak butuh PyYAML.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Muat file config (.yaml/.yml/.json) menjadi dict."""
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML dibutuhkan untuk config YAML: pip install pyyaml"
            ) from e
        cfg = yaml.safe_load(text)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config {p} harus berupa mapping YAML, bukan {type(cfg)}")
        return cfg
    return json.loads(text)


def config_repr(cfg: dict[str, Any]) -> str:
    """Representasi Python literal (json.dumps) untuk dibenamkan di sel notebook."""
    return json.dumps(cfg, indent=4, ensure_ascii=False)


def config_snippet(cfg: dict[str, Any], var_name: str = "CONFIG") -> str:
    """Source untuk sel Config: `CONFIG = {...}`."""
    return f"{var_name} = {config_repr(cfg)}"
