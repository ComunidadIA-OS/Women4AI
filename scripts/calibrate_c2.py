"""Calibra τ (umbral de coseno de C2) contra el gold set. No se fija a ojo.

    python scripts/calibrate_c2.py [--tol 1.0] [--min 0.50 --max 0.99 --step 0.01]

Necesita, por episodio anotado: ``data/annotations/<ep>/events.json`` (con eventos
``C2_reformulation``) y ``data/interim/<ep>/turns.parquet`` (pipeline ejecutado hasta
``turns``). Calcula los embeddings MrBERT-es una vez, barre τ y elige el de mayor F1 (en
empate, el más alto). Escribe la curva en ``data/processed/c2_calibration.json`` y muestra
la línea que hay que poner en ``configs/default.yaml``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from explicometro import io as aio
from explicometro.config import load_config
from explicometro.evaluation import best_threshold, sweep_threshold
from explicometro.events import EventsParams, MrBertEmbedder, c2_candidates, reformulations_from
from explicometro.report import fmt3
from explicometro.schemas import Event, EventType, Turn


def main(argv: list[str] | None = None, embedder_factory: Callable | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--tol", type=float, default=1.0, help="Tolerancia temporal al emparejar con el gold (s)")
    ap.add_argument("--min", type=float, default=0.50)
    ap.add_argument("--max", type=float, default=0.99)
    ap.add_argument("--step", type=float, default=0.01)
    a = ap.parse_args(argv)
    cfg = load_config(a.config if a.config.exists() else None)
    rcfg = cfg.thresholds.reformulation

    episodes, n_gold = [], 0
    embed = None
    ev_params = cfg.stages.events.parsed(EventsParams)
    for gold_path in sorted(cfg.paths.annotations.glob("*/events.json")):
        ep = gold_path.parent.name
        gold = [g for g in aio.read_models(gold_path, Event) if g.type == EventType.REFORMULATION]
        turns_path = cfg.paths.interim / ep / "turns.parquet"
        if not gold or not turns_path.exists():
            print(f"[{ep}] sin C2 en el gold o sin turns.parquet: se omite")
            continue
        if embed is None:  # el modelo se carga solo si hay algo que calibrar
            embed = embedder_factory() if embedder_factory else MrBertEmbedder(ev_params, cfg.runtime)
        cands = c2_candidates(aio.read_parquet(turns_path, Turn), embed, rcfg)
        episodes.append((lambda tau, c=cands: reformulations_from(c, tau), gold))
        n_gold += len(gold)
    if not episodes:
        print("No hay episodios con C2 anotado: no se puede calibrar τ.")
        return 1

    taus = [float(t) for t in np.round(np.arange(a.min, a.max + 1e-9, a.step), 4)]
    curve = sweep_threshold(episodes, taus, EventType.REFORMULATION, a.tol)
    best = best_threshold(curve)
    print(f"{len(episodes)} episodios, {n_gold} C2 en el gold")
    print(f"{'τ':>6} {'P':>6} {'R':>6} {'F1':>6}  tp/fp/fn")
    for c in curve[:: max(1, round(0.05 / a.step))]:
        prf = f"{fmt3(c['precision'])} {fmt3(c['recall'])} {fmt3(c['f1'])}"
        print(f"{c['tau']:>6.2f} {prf}  {c['tp']}/{c['fp']}/{c['fn']}")
    out = cfg.paths.processed / "c2_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"n_episodes": len(episodes), "n_gold": n_gold, "tol_s": a.tol, "best": best, "curve": curve},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if best is None:
        print("Ningún τ da aciertos: revisa el gold o el modelo de embeddings.")
        return 1
    if n_gold < 20:
        print(f"AVISO: solo {n_gold} C2 en el gold; τ poco fiable. Anota más antes de publicar.")
    print(f"\nMejor τ = {best['tau']} (F1 {best['f1']:.3f}). En configs/default.yaml:")
    print(f"  thresholds.reformulation.cos_threshold: {best['tau']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
