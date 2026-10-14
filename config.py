"""
Carga de configuracion central global con soporte de overrides por CLI y demo.

Provee acceso unificado a los parametros de Parte 1 (Optimizacion Numerica)
y Parte 2 (Optimizacion Combinatoria - TSP 47 Capitales).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
CONFIG_POR_DEFECTO = BASE_DIR / "configuracion.json"


def _aplicar_demo(base_dict: dict, demo_dict: dict) -> dict:
    """Aplica recursivamente los overrides de demo sobre la configuracion base."""
    res = copy.deepcopy(base_dict)
    for k, v in demo_dict.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(res.get(k), dict):
            res[k] = _aplicar_demo(res[k], v)
        else:
            res[k] = v
    return res


def cargar_config(
    ruta: str | Path | None = None,
    demo: bool = False,
    parte: str | int | None = None,
) -> dict[str, Any]:
    """Carga configuracion.json y aplica el perfil demo si se solicita.

    Parametros:
      ruta: Ruta personalizada a un archivo JSON. Si es None, busca configuracion.json.
      demo: Si True, sobreescribe con los valores del bloque 'demo'.
      parte: 1 o 'parte_1' para obtener solo Parte 1; 2 o 'parte_2' para Parte 2; None para todo.
    """
    if ruta is None:
        if CONFIG_POR_DEFECTO.exists():
            ruta = CONFIG_POR_DEFECTO
        else:
            # Buscar en directorio padre por si se ejecuta desde subcarpetas
            padre = Path.cwd().parent / "configuracion.json"
            if padre.exists():
                ruta = padre
            else:
                ruta = CONFIG_POR_DEFECTO

    ruta = Path(ruta)
    with open(ruta, encoding="utf-8") as f:
        cfg = json.load(f)

    if demo and "demo" in cfg:
        demo_cfg = cfg["demo"]
        if "parte_1" in cfg and "parte_1" in demo_cfg:
            cfg["parte_1"] = _aplicar_demo(cfg["parte_1"], demo_cfg["parte_1"])
        if "parte_2" in cfg and "parte_2" in demo_cfg:
            cfg["parte_2"] = _aplicar_demo(cfg["parte_2"], demo_cfg["parte_2"])

    if parte in (1, "1", "parte_1"):
        return cfg.get("parte_1", {})
    elif parte in (2, "2", "parte_2"):
        return cfg.get("parte_2", {})

    return cfg
