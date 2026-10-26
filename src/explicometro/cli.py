"""CLI del Explicómetro.

    explicometro run --episode X [--until align] [--audio ruta] [--dry-run] [--force]
    explicometro run --episode X --only diarize --force
    explicometro validate data/interim/X/metrics.json

Con ``--episode`` y sin ``--audio`` busca ``data/raw/X.<ext>`` (paths.raw del YAML).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from .config import load_config
from .pipeline import STAGES, Pipeline, upstream
from .schemas import EpisodeMetrics


def episode_id_from(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "episode"


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".mp4", ".mkv", ".webm")


def find_source(raw_dir: Path, episode: str) -> Path | None:
    """Primer fichero de audio/vídeo en ``raw_dir`` cuyo nombre (sin extensión) es ``episode``."""
    if not raw_dir.exists():
        return None
    hits = sorted(p for p in raw_dir.iterdir() if p.stem == episode and p.suffix.lower() in AUDIO_EXTS)
    return hits[0] if hits else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="explicometro", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Ejecuta el pipeline sobre un episodio")
    r.add_argument("--audio", type=Path, help="Audio de entrada (obligatorio la primera vez)")
    r.add_argument("--episode", help="ID del episodio (por defecto, nombre del fichero)")
    r.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    r.add_argument("--dry-run", action="store_true", help="CPU, 120 s, stubs en etapas con modelos")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--until", choices=STAGES, help="Ejecuta hasta esta etapa incluida (y sus deps)")
    g.add_argument("--only", nargs="+", choices=STAGES, help="Ejecuta solo estas etapas (+deps desde caché)")
    r.add_argument(
        "--force", action="store_true", help="Recalcula: con --until, toda la cadena; con --only, esas etapas"
    )
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
    source = args.audio or find_source(cfg.paths.raw, episode)
    if source is not None and not source.exists():
        print(f"No existe {source}", file=sys.stderr)
        return 2
    pipe = Pipeline(cfg=cfg, episode_id=episode, source=source, dry_run=args.dry_run)
    if args.only:
        targets, force = args.only, args.only if args.force else []
    else:
        targets = [args.until] if args.until else list(STAGES)
        force = upstream(targets) if args.force else []
    pipe.run(targets, force=force)
    print(f"Artefactos en {pipe.workdir} (tiempos en timings.jsonl, log en pipeline.log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
