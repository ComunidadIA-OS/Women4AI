"""Entrena el clasificador de género de voz (ECAPA + LogisticRegression) con Common Voice.

    python scripts/train_gender.py                      # usa configs/gender_train.yaml
    python scripts/train_gender.py --skip-download      # solo con archivos ya descargados

Pasos: descarga (Mozilla Data Collective) → lectura de validated.tsv en streaming →
selección equilibrada (máx. 2.000 clips/clase, tope por hablante) → split POR HABLANTE →
extracción solo de los clips elegidos → embeddings ECAPA (con caché) → LogisticRegression →
reporte (accuracy, F1 por clase, matriz de confusión; por clip, por hablante y con umbral)
→ models/gender_lr.joblib + models/gender_lr.report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from explicometro import gender_training as gt
from explicometro.gender import EcapaEmbedder, l2norm

log = logging.getLogger("train_gender")
SR = 16000


def decode(path: Path, max_s: float) -> np.ndarray:
    """mp3/wav → float32 16 kHz mono vía ffmpeg (sin ficheros intermedios)."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-t", str(max_s),
           "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]  # fmt: skip
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def resolve_source(src: dict[str, Any], download_dir: Path, skip_download: bool) -> Path | None:
    """Ruta a un .tar.gz o a un directorio ya extraído de Common Voice."""
    if src.get("archive"):
        return Path(src["archive"])
    if not src.get("dataset_id"):
        log.warning("%s: sin dataset_id ni archive; se omite", src["locale"])
        return None
    if skip_download:
        hits = sorted(download_dir.glob("*.tar.gz"))
        log.warning(
            "%s: --skip-download; indica 'archive' en el YAML (hay %d .tar.gz)", src["locale"], len(hits)
        )
        return None
    try:
        from datacollective import download_dataset
    except ImportError:
        raise SystemExit("Falta el SDK de MDC: pip install -e '.[train]'") from None
    log.info(
        "Descargando %s (%s) en %s. Puede ser muy grande.", src["locale"], src["dataset_id"], download_dir
    )
    return Path(download_dataset(src["dataset_id"], download_directory=str(download_dir)))


def read_tsv(source: Path, locale: str) -> pd.DataFrame:
    if source.is_dir():
        hits = sorted(source.rglob(f"{locale}/validated.tsv"))
        if not hits:
            raise SystemExit(f"No hay {locale}/validated.tsv bajo {source}")
        return gt.parse_cv_tsv(hits[0], locale)
    tsvs = gt.tsv_members(source, [locale])
    if locale not in tsvs:
        raise SystemExit(f"{source} no contiene {locale}/validated.tsv")
    return gt.parse_cv_tsv(tsvs[locale], locale)


def clip_path(source: Path, clips_dir: Path, locale: str, name: str) -> Path:
    if source.is_dir():
        hits = list(source.rglob(f"{locale}/clips/{name}"))
        return hits[0] if hits else source / locale / "clips" / name
    return clips_dir / locale / name


def embed_all(
    sel: pd.DataFrame, paths: list[Path], embed: Callable[[list[np.ndarray]], np.ndarray], cache: Path,
    max_s: float, chunk: int = 256,
) -> np.ndarray:  # fmt: skip
    """Embeddings por clip con caché en parquet (clave: locale/path)."""
    keys = (sel["locale"] + "/" + sel["path"]).tolist()
    done: dict[str, np.ndarray] = {}
    if cache.exists():
        old = pd.read_parquet(cache)
        done = {k: np.asarray(e, dtype=np.float32) for k, e in zip(old["key"], old["emb"], strict=True)}
    todo = [i for i, k in enumerate(keys) if k not in done]
    for start in range(0, len(todo), chunk):
        idx = todo[start : start + chunk]
        embs = embed([decode(paths[i], max_s) for i in idx])
        for i, e in zip(idx, embs, strict=True):
            done[keys[i]] = e
        log.info("embeddings %d/%d", min(start + chunk, len(todo)), len(todo))
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"key": list(done), "emb": [v.tolist() for v in done.values()]}).to_parquet(cache)
    return l2norm(np.stack([done[k] for k in keys]))


def main(argv: list[str] | None = None, embedder_factory: Callable[[dict], Callable] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/gender_train.yaml"))
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(override=False)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    work, dl = Path(cfg["work_dir"]), Path(cfg["download_dir"])

    sources: dict[str, Path] = {}
    for src in cfg["sources"]:
        path = resolve_source(src, dl, a.skip_download)
        if path is not None:
            sources[src["locale"]] = path
    if not sources:
        raise SystemExit("Ninguna fuente disponible: rellena dataset_id o archive en el YAML")

    df = pd.concat([read_tsv(p, loc) for loc, p in sources.items()], ignore_index=True)
    log.info("Common Voice con género H/M: %s", df.groupby(["locale", "gender"]).size().to_dict())
    sel = gt.speaker_split(
        gt.select_balanced(df, cfg["per_class"], cfg["max_per_speaker"], cfg["seed"]),
        cfg["test_frac"],
        cfg["seed"],
    )
    work.mkdir(parents=True, exist_ok=True)
    sel.to_csv(work / "selection.csv", index=False)

    for loc, p in sources.items():
        if not p.is_dir():
            wanted = {loc: set(sel.loc[sel.locale == loc, "path"])}
            log.info(
                "%s: extrayendo %d clips de %s", loc, gt.extract_clips(p, wanted, work / "clips"), p.name
            )
    paths = [clip_path(sources[r.locale], work / "clips", r.locale, r.path) for r in sel.itertuples()]

    embed = (embedder_factory or (lambda c: EcapaEmbedder(c["embedding_model"], c["embedding_savedir"],
                                                          c["device"], c["batch_size"])))(cfg)  # fmt: skip
    X = embed_all(sel, paths, embed, work / "embeddings.parquet", cfg["max_clip_s"])
    y, groups = sel["gender"].to_numpy(), sel["client_id"].to_numpy()
    tr, te = (sel["split"] == "train").to_numpy(), (sel["split"] == "test").to_numpy()

    model = gt.train(X[tr], y[tr], cfg["C"], cfg["seed"])
    report = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {
            "sources": {
                loc: (str(s.get("dataset_id")) if s.get("dataset_id") else str(sources[loc]))
                for s in cfg["sources"]
                for loc in [s["locale"]]
                if loc in sources
            },
            "clips": sel.groupby(["split", "locale", "gender"]).size().reset_index().to_dict("records"),
            "speakers": {k: int(v) for k, v in sel.groupby("split")["client_id"].nunique().items()},
        },
        "test": gt.evaluate(model, X[te], y[te], groups[te], cfg["min_conf"]),
        "train": gt.evaluate(model, X[tr], y[tr], groups[tr], cfg["min_conf"]),
    }
    import joblib
    import sklearn

    out = Path(cfg["out_model"])
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "classes": gt.CLASSES, "embedding_model": cfg["embedding_model"],
         "sklearn": sklearn.__version__, "report": report},
        out,
    )  # fmt: skip
    Path(cfg["out_report"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("== TEST (hablantes no vistos) ==")
    print(gt.format_report(report["test"]))
    print(f"Modelo: {out} · reporte: {cfg['out_report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
