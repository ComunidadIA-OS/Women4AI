"""Entrenamiento y evaluación del clasificador de género de voz (funciones puras + IO mínima).

Datos: Common Voice castellano y catalán (campo ``gender``), descargado de Mozilla Data
Collective. Claves para no engañarnos con las métricas:

- **Split por hablante** (``client_id``): ningún hablante está en train y test a la vez.
- **Tope de clips por hablante** para que unas pocas voces prolíficas no dominen.
- Se evalúa por clip y **por hablante** (media de embeddings, como en inferencia) y con el
  umbral de confianza, informando de la cobertura (% que no queda en ``U``).
- Solo se usan contribuciones con género declarado masculino/femenino; el resto
  (no binario, intersexual, trans, no declarado, vacío) no entra en un clasificador binario.
"""

from __future__ import annotations

import csv
import io
import logging
import tarfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gender import l2norm, speaker_embedding
from .schemas import Gender

log = logging.getLogger(__name__)

CV_GENDER = {
    "male": Gender.MALE.value,
    "male_masculine": Gender.MALE.value,
    "female": Gender.FEMALE.value,
    "female_feminine": Gender.FEMALE.value,
}
CLASSES = [Gender.MALE.value, Gender.FEMALE.value]


# ----------------------------------------------------------------------- Common Voice


def parse_cv_tsv(data: bytes | str | Path, locale: str) -> pd.DataFrame:
    """``validated.tsv`` → ``client_id, path, gender (H/M), locale``. Descarta género no binario
    o no declarado. ``QUOTE_NONE``: las frases llevan comillas sin escapar."""
    src = io.BytesIO(data) if isinstance(data, bytes) else data
    df = pd.read_csv(src, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
                     usecols=lambda c: c in {"client_id", "path", "gender"})  # fmt: skip
    if not {"client_id", "path", "gender"} <= set(df.columns):
        raise ValueError(f"TSV sin columnas client_id/path/gender: {list(df.columns)}")
    df["gender"] = df["gender"].str.strip().str.lower().map(CV_GENDER)
    df = df.dropna(subset=["gender"])
    df = df[(df["client_id"] != "") & (df["path"] != "")]
    return df.assign(locale=locale)[["client_id", "path", "gender", "locale"]].reset_index(drop=True)


def select_balanced(
    df: pd.DataFrame, per_class: int = 2000, max_per_speaker: int = 5, seed: int = 13
) -> pd.DataFrame:
    """Subconjunto equilibrado: ``per_class`` clips por género, repartidos a partes iguales
    entre idiomas, con como mucho ``max_per_speaker`` clips de cada hablante."""
    rng = np.random.default_rng(seed)
    locales = sorted(df["locale"].unique())
    capped = (
        df.sample(frac=1.0, random_state=seed)
        .groupby(["locale", "client_id"], group_keys=False)
        .head(max_per_speaker)
    )
    parts = []
    for gender in CLASSES:
        quota = {
            loc: per_class // len(locales) + (i < per_class % len(locales)) for i, loc in enumerate(locales)
        }
        for loc in locales:
            pool = capped[(capped["gender"] == gender) & (capped["locale"] == loc)]
            take = min(quota[loc], len(pool))
            if take < quota[loc]:
                log.warning("Common Voice %s/%s: solo %d clips de %d", loc, gender, take, quota[loc])
            parts.append(pool.iloc[rng.permutation(len(pool))[:take]])
    return pd.concat(parts, ignore_index=True)


def speaker_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 13) -> pd.DataFrame:
    """Añade ``split`` (train/test) asignando **hablantes completos** (``client_id``, que es
    único en todo Common Voice: una persona que graba en es y ca va entera a un lado),
    estratificado por género e idioma principal del hablante."""
    rng = np.random.default_rng(seed)
    first = df.groupby("client_id").agg(gender=("gender", "first"), locale=("locale", "first"))
    test_ids: set[str] = set()
    for _key, grp in first.groupby(["gender", "locale"]):
        ids = sorted(grp.index)
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * test_frac)) if len(ids) > 1 else 0
        test_ids |= set(ids[:n_test])
    out = df.assign(split=np.where(df["client_id"].isin(test_ids), "test", "train"))
    leak = set(out.loc[out.split == "train", "client_id"]) & set(out.loc[out.split == "test", "client_id"])
    assert not leak, f"fuga de hablantes entre train y test: {len(leak)}"
    return out


def tsv_members(archive: Path, locales: Iterable[str]) -> dict[str, bytes]:
    """Pasada 1 sobre el ``.tar.gz`` en streaming: devuelve ``validated.tsv`` de cada idioma."""
    wanted = set(locales)
    out: dict[str, bytes] = {}
    with tarfile.open(archive, "r|gz") as tar:
        for m in tar:
            parts = Path(m.name).parts
            if m.isfile() and parts[-1] == "validated.tsv" and len(parts) >= 2 and parts[-2] in wanted:
                out[parts[-2]] = tar.extractfile(m).read()  # type: ignore[union-attr]
    return out


def extract_clips(archive: Path, wanted: dict[str, set[str]], dest: Path) -> int:
    """Pasada 2: extrae solo los clips elegidos a ``dest/<locale>/<fichero>``.

    No usa ``extractall``: el destino lo construimos nosotros (sin rutas del archivo)."""
    n = 0
    with tarfile.open(archive, "r|gz") as tar:
        for m in tar:
            parts = Path(m.name).parts
            if not m.isfile() or len(parts) < 3 or parts[-2] != "clips":
                continue
            loc, name = parts[-3], parts[-1]
            if name in wanted.get(loc, ()):
                out = dest / loc / name
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    out.write_bytes(tar.extractfile(m).read())  # type: ignore[union-attr]
                n += 1
    return n


# ----------------------------------------------------------------- modelo y métricas


def train(X: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 13) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed),
    )
    return model.fit(l2norm(X), y)


def _block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    f1 = f1_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4) if len(y_true) else None,
        "f1": {c: round(float(v), 4) for c, v in zip(CLASSES, f1, strict=True)},
        "confusion": {"labels": CLASSES, "matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()},
    }


def evaluate(
    model: Any, X: np.ndarray, y: np.ndarray, groups: np.ndarray, min_conf: float = 0.8
) -> dict[str, Any]:
    """Métricas por clip y por hablante; con umbral, cobertura y métricas sobre lo cubierto
    (lo que queda en U se excluye, como en el pipeline, y se informa cuánto es)."""
    X = l2norm(X)
    proba = model.predict_proba(X)
    pred = model.classes_[proba.argmax(1)]
    report: dict[str, Any] = {"clip": _block(y, pred)}

    spk_ids = sorted(set(groups))
    Xs = np.stack([speaker_embedding(X[groups == s]) for s in spk_ids])
    ys = np.array([y[groups == s][0] for s in spk_ids])
    ps = model.predict_proba(Xs)
    pred_s = model.classes_[ps.argmax(1)]
    report["speaker"] = _block(ys, pred_s)

    conf = ps.max(1)
    covered = conf >= min_conf
    report["speaker_thresholded"] = {
        "min_conf": min_conf,
        "coverage": round(float(covered.mean()), 4) if len(covered) else None,
        "unknown": int((~covered).sum()),
        **_block(ys[covered], pred_s[covered]),
    }
    return report


def format_report(r: dict[str, Any]) -> str:
    lines = []
    for name in ("clip", "speaker", "speaker_thresholded"):
        b = r[name]
        cm = b["confusion"]["matrix"]
        extra = f" · cobertura {b['coverage']:.1%} ({b['unknown']} → U)" if "coverage" in b else ""
        lines += [
            f"[{name}] n={b['n']} accuracy={b['accuracy']} F1 H={b['f1']['H']} M={b['f1']['M']}{extra}",
            f"    matriz (filas=verdad, cols=pred; H, M): {cm[0]} / {cm[1]}",
        ]
    return "\n".join(lines)
