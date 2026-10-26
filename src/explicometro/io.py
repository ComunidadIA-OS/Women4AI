"""Lectura/escritura de artefactos. Solo serializa tipos de ``schemas``.

Formatos: JSON (listas de modelos o un modelo), RTTM (diarización), Parquet (tablas planas).
Las escrituras son atómicas (fichero temporal + rename) para que la caché nunca vea
artefactos a medio escribir.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, TypeAdapter

from .schemas import Segment

M = TypeVar("M", bound=BaseModel)


@contextmanager
def _atomic(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        yield tmp_path
        tmp_path.chmod(0o644)  # mkstemp crea 0600; los artefactos son de lectura compartida
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_text(path: Path, text: str) -> None:
    with _atomic(path) as tmp:
        tmp.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------- JSON


def write_model(path: Path, model: BaseModel) -> None:
    write_text(path, model.model_dump_json(indent=2) + "\n")


def read_model(path: Path, cls: type[M]) -> M:
    return cls.model_validate_json(path.read_text(encoding="utf-8"))


def write_models(path: Path, models: Sequence[BaseModel]) -> None:
    data = [m.model_dump(mode="json") for m in models]
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_models(path: Path, cls: type[M]) -> list[M]:
    return TypeAdapter(list[cls]).validate_json(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- RTTM


def write_rttm(path: Path, segments: Iterable[Segment], episode_id: str) -> None:
    """RTTM estándar: ``SPEAKER <file> 1 <onset> <dur> <NA> <NA> <spk> <NA> <NA>``."""
    lines = []
    for s in segments:
        if s.speaker is None:
            raise ValueError("RTTM requiere segmentos con speaker")
        lines.append(f"SPEAKER {episode_id} 1 {s.start:.3f} {s.duration:.3f} <NA> <NA> {s.speaker} <NA> <NA>")
    write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_rttm(path: Path) -> list[Segment]:
    out: list[Segment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "SPEAKER":
            continue
        onset, dur, spk = float(parts[3]), float(parts[4]), parts[7]
        out.append(Segment(start=onset, end=round(onset + dur, 3), speaker=spk))
    return out


# --------------------------------------------------------------------------- Parquet


def write_parquet(path: Path, models: Sequence[BaseModel], cls: type[BaseModel]) -> None:
    """Tabla plana; con lista vacía conserva las columnas del esquema."""
    rows = [m.model_dump(mode="json") for m in models]
    df = pd.DataFrame(rows, columns=list(cls.model_fields))
    with _atomic(path) as tmp:
        df.to_parquet(tmp, index=False)


def read_parquet(path: Path, cls: type[M]) -> list[M]:
    df = pd.read_parquet(path)
    records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    return [cls.model_validate(r) for r in records]
