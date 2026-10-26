"""Compara eventos predichos (data/interim/<ep>/events.parquet) con anotaciones.

Anotaciones: ``data/annotations/<ep>/events.json`` = lista de ``Event`` (mismo esquema).
Sin anotaciones, termina con código 0 e informa de que no hay nada que evaluar.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from explicometro import io as aio
from explicometro.config import load_config
from explicometro.evaluation import PRF, score_events
from explicometro.schemas import Event


def _fmt(x: float | None) -> str:
    return "  —  " if x is None else f"{x:.3f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--tol", type=float, default=1.0, help="Tolerancia de inicio (s)")
    a = p.parse_args()
    cfg = load_config(a.config if a.config.exists() else None)

    gold_files = sorted(cfg.paths.annotations.glob("*/events.json"))
    if not gold_files:
        print(f"Sin anotaciones en {cfg.paths.annotations}/<episodio>/events.json; nada que evaluar.")
        return 0
    total: dict[str, PRF] = defaultdict(PRF)
    for gf in gold_files:
        ep = gf.parent.name
        pred_path = cfg.paths.interim / ep / "events.parquet"
        if not pred_path.exists():
            print(f"[{ep}] sin predicciones ({pred_path}); se omite")
            continue
        scores = score_events(aio.read_parquet(pred_path, Event), aio.read_models(gf, Event), a.tol)
        for etype, s in scores.items():
            t = total[etype.value]
            t.tp, t.fp, t.fn = t.tp + s.tp, t.fp + s.fp, t.fn + s.fn
    print(f"{'tipo':32} {'P':>6} {'R':>6} {'F1':>6}   tp/fp/fn")
    for name, s in sorted(total.items()):
        print(f"{name:32} {_fmt(s.precision)} {_fmt(s.recall)} {_fmt(s.f1)}   {s.tp}/{s.fp}/{s.fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
