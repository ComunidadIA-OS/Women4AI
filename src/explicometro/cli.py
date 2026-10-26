"""CLI: ``explicometro run --audio X [--dry-run] [--only STAGE ...] [--force]``."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from .config import load_config
from .pipeline import STAGES, Pipeline
from .schemas import EpisodeMetrics


def episode_id_from(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "episode"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="explicometro", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Ejecuta el pipeline sobre un episodio")
    r.add_argument("--audio", type=Path, help="Audio de entrada (obligatorio la primera vez)")
    r.add_argument("--episode", help="ID del episodio (por defecto, nombre del fichero)")
    r.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    r.add_argument("--dry-run", action="store_true", help="CPU, 120 s, stubs en etapas con modelos")
    r.add_argument("--only", nargs="+", choices=STAGES, help="Ejecuta solo estas etapas (+deps)")
    r.add_argument("--force", action="store_true", help="Ignora la caché de las etapas pedidas")
    r.add_argument("-v", "--verbose", action="store_true")
    v = sub.add_parser("validate", help="Valida un metrics.json contra el esquema")
    v.add_argument("path", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.cmd == "validate":
        m = EpisodeMetrics.model_validate_json(args.path.read_text(encoding="utf-8"))
        print(f"OK {args.path} (episode={m.episode_id}, schema={m.schema_version})")
        return 0

    if args.audio is None and args.episode is None:
        print("Indica --audio o --episode", file=sys.stderr)
        return 2
    episode = args.episode or episode_id_from(args.audio)
    cfg = load_config(args.config if args.config.exists() else None, dry_run=args.dry_run)
    pipe = Pipeline(cfg=cfg, episode_id=episode, source=args.audio, dry_run=args.dry_run)
    targets = args.only or list(STAGES)
    pipe.run(targets, force=targets if args.force else ())
    print(f"Artefactos en {pipe.workdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
