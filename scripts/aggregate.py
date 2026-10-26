"""Agrega métricas de episodios por corpus, cadena, año (o lo que haya en el manifiesto).

    python scripts/aggregate.py                          # corpus + channel + year + channel,year
    python scripts/aggregate.py --by channel --by program,year

Lee ``data/interim/*/metrics.json`` (sin dry-runs) y los cruza con ``configs/episodes.yaml``
(campos ``channel``, ``program``, ``year``...). Escribe en ``data/processed/``:
``metrics_<agrupación>.json`` (lista de GroupMetrics), ``metrics_<agrupación>.csv`` (tabla
plana) y ``report_<agrupación>.md``. IC 95 % por bootstrap remuestreando episodios.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from explicometro import io as aio
from explicometro.config import load_config
from explicometro.metrics import aggregate
from explicometro.report import to_markdown
from explicometro.schemas import EpisodeMetrics, GroupMetrics

log = logging.getLogger("aggregate")
DEFAULT_GROUPINGS = [[], ["channel"], ["year"], ["channel", "year"]]


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    eps = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("episodes", [])
    return {str(e["id"]): {k: str(v) for k, v in e.items() if v is not None} for e in eps}


def group_episodes(
    eps: list[EpisodeMetrics], meta: dict[str, dict[str, str]], keys: list[str]
) -> dict[tuple[str, ...], list[EpisodeMetrics]]:
    groups: dict[tuple[str, ...], list[EpisodeMetrics]] = defaultdict(list)
    for e in eps:
        m = meta.get(e.episode_id, {})
        groups[tuple(m.get(k, "desconocido") for k in keys)].append(e)
    return dict(sorted(groups.items()))


def flat_row(g: GroupMetrics) -> dict[str, object]:
    row: dict[str, object] = {**g.group, "n_episodes": g.n_episodes, "hours": round(g.duration_s / 3600, 2)}
    for gen, s in g.presence.by_gender.items():
        row |= {
            f"A_speakers_{gen.value}": s.speakers,
            f"A_time_{gen.value}": s.time,
            f"A_words_{gen.value}": s.words,
        }
    row |= {"U_time_share": g.unknown.time_share}
    for gen, n in g.protagonism.n_turns.items():
        row |= {f"B_turns_{gen.value}": n, f"B_mean_turn_s_{gen.value}": g.protagonism.mean_turn_s.get(gen)}
    for etype, dyads in g.events.items():
        for d, s in dyads.items():
            row |= {f"{etype.value}_{d.value}_n": s.count, f"{etype.value}_{d.value}_rate": s.rate_per_min}
    for name, r in (("excess", g.interruption_excess), ("asymmetry", g.asymmetry_ratio)):
        row |= {name: r.value, f"{name}_ci_low": r.ci_low, f"{name}_ci_high": r.ci_high,
                f"{name}_n": r.n_events, f"{name}_status": r.status}  # fmt: skip
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--manifest", type=Path, default=Path("configs/episodes.yaml"))
    ap.add_argument("--by", action="append", help="Claves separadas por comas; repetible. Por defecto: "
                    "corpus, channel, year, channel+year")  # fmt: skip
    ap.add_argument("--out", type=Path, default=None, help="Por defecto paths.processed")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(a.config if a.config.exists() else None)
    out = a.out or cfg.paths.processed

    eps = [aio.read_model(p, EpisodeMetrics) for p in sorted(cfg.paths.interim.glob("*/metrics.json"))]
    eps = [e for e in eps if not e.dry_run]
    if not eps:
        print(f"No hay metrics.json (no dry-run) en {cfg.paths.interim}")
        return 0
    meta = load_manifest(a.manifest)
    missing = [e.episode_id for e in eps if e.episode_id not in meta]
    if missing:
        log.warning(
            "%d episodios sin entrada en %s → 'desconocido': %s", len(missing), a.manifest, missing[:5]
        )

    groupings = [[k for k in by.split(",") if k] for by in a.by] if a.by else DEFAULT_GROUPINGS
    out.mkdir(parents=True, exist_ok=True)
    for keys in groupings:
        name = "_".join(keys) or "corpus"
        results = [
            aggregate(group, cfg.thresholds, keys, dict(zip(keys, values, strict=True)))
            for values, group in group_episodes(eps, meta, keys).items()
        ]
        payload = json.dumps([r.model_dump(mode="json") for r in results], ensure_ascii=False, indent=2)
        aio.write_text(out / f"metrics_{name}.json", payload + "\n")
        pd.DataFrame([flat_row(r) for r in results]).to_csv(out / f"metrics_{name}.csv", index=False)
        aio.write_text(out / f"report_{name}.md", "\n\n---\n\n".join(to_markdown(r) for r in results))
        print(f"{name}: {len(results)} grupos → {out}/metrics_{name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
