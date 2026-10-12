"""Carga de configuracion.json con overrides por CLI.

Todos los experimentos leen este archivo para no editar codigo fuente.
Uso: cfg = cargar_config(args.config, demo=args.demo)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CONFIG_POR_DEFECTO = BASE_DIR / "configuracion.json"


def cargar_config(ruta: str | Path | None = None, demo: bool = False) -> dict:
    ruta = Path(ruta) if ruta else CONFIG_POR_DEFECTO
    with open(ruta, encoding="utf-8") as f:
        cfg = json.load(f)
    if demo and "demo" in cfg:
        d = cfg["demo"]
        cfg = copy.deepcopy(cfg)
        for clave in ("valor_hora", "ga", "aco", "sensibilidad", "gif"):
            if clave in d:
                if isinstance(d[clave], dict) and isinstance(cfg.get(clave), dict):
                    cfg[clave] = {**cfg[clave], **d[clave]}
                else:
                    cfg[clave] = d[clave]
    return cfg
