"""
Modulo de configuracion local para Parte 2 (Optimizacion Combinatoria - TSP).
Carga directamente la configuracion central global sin colisiones de importacion.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PY = ROOT_DIR / "config.py"

_spec = importlib.util.spec_from_file_location("global_config", CONFIG_PY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def cargar_config(ruta=None, demo: bool = False, parte: str | int | None = "parte_2") -> dict:
    return _mod.cargar_config(ruta=ruta, demo=demo, parte="parte_2")
