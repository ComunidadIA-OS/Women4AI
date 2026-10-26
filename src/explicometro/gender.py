"""Género por hablante, con tres fuentes por orden de prioridad.

1. **metadata**: ``data/annotations/speakers_<episodio>.csv`` (``spk_id, nombre_publico,
   genero_publico``). Fuente preferente: es la identidad pública, no una inferencia.
   - ``genero_publico`` vacío → se pasa al clasificador.
   - ``H``/``M`` (o hombre/mujer, male/female...) → se usa tal cual (conf 1.0).
   - ``U``/no binario/no consta... → ``U`` explícito: **no** se clasifica por la voz.
2. **classifier**: embeddings ECAPA (speechbrain/spkrec-ecapa-voxceleb) de ventanas de voz
   limpia (sin solapes con otros hablantes), promediados por hablante, + LogisticRegression
   (``models/gender_lr.joblib``, entrenado con ``scripts/train_gender.py``).
3. **unknown**: si la voz limpia es < ``min_clean_s`` (30 s) o la confianza < ``min_conf`` (0,8).

La etiqueta es una inferencia binaria a partir de la voz, no la identidad de nadie: ver
docs/gender.md. Las métricas excluyen ``U`` y reportan cuánto tiempo queda fuera.
"""

from __future__ import annotations

import csv
import logging
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from .audio import read_window
from .backends import BackendError, resolve
from .config import Params, RuntimeCfg, StageCfg
from .schemas import AudioMeta, Gender, GenderSource, Segment, Speaker

log = logging.getLogger(__name__)

Embedder = Callable[[list[np.ndarray]], np.ndarray]


class GenderParams(Params):
    metadata_csv: str = Field(default="speakers_{episode}.csv", description="Relativo a paths.annotations")
    classifier_path: str = "models/gender_lr.joblib"
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    embedding_savedir: str = "models/pretrained/spkrec-ecapa-voxceleb"
    min_clean_s: float = Field(default=30.0, ge=0)
    min_conf: float = Field(default=0.8, ge=0.5, le=1.0)
    window_s: float = Field(default=4.0, gt=0)
    min_window_s: float = Field(default=1.0, gt=0)
    max_windows: int = Field(default=120, ge=1)
    overlap_guard_s: float = Field(default=0.1, ge=0, description="Margen extra alrededor de solapes")
    batch_size: int = Field(default=16, ge=1)
    device_index: int = Field(default=0, ge=0)


# ------------------------------------------------------------------------------ metadata

_H = {"h", "hombre", "home", "masculino", "male", "male_masculine", "man"}
_M = {"m", "f", "mujer", "dona", "femenino", "female", "female_feminine", "woman"}  # M = mujer
_U = {"u", "unknown", "desconocido", "no consta", "nc", "no binario", "no binaria", "non-binary",
      "nonbinary", "no binari", "otro", "otra", "x"}  # fmt: skip


_ROLE = {"moderador": "moderator", "moderadora": "moderator", "presentador": "moderator",
         "presentadora": "moderator", "conductor": "moderator", "conductora": "moderator",
         "moderator": "moderator", "invitado": "guest", "invitada": "guest", "convidat": "guest",
         "convidada": "guest", "tertuliano": "guest", "tertuliana": "guest", "guest": "guest"}  # fmt: skip


@dataclass(frozen=True)
class MetaRow:
    public_name: str | None
    gender: Gender | None  # None = vacío → clasificador
    role: str | None = None  # "moderator" | "guest" | None (columna opcional ``rol``)


def parse_role(value: str) -> str | None:
    v = _norm(value)
    if not v:
        return None
    if v not in _ROLE:
        raise ValueError(f"rol no reconocido: {value!r} (usa moderador/a, invitado/a o déjalo vacío)")
    return _ROLE[v]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip().lower())


def parse_gender(value: str) -> Gender | None:
    v = _norm(value)
    if not v:
        return None
    if v in _H:
        return Gender.MALE
    if v in _M:
        return Gender.FEMALE
    if v in _U:
        return Gender.UNKNOWN
    raise ValueError(f"genero_publico no reconocido: {value!r} (usa H, M, U o déjalo vacío)")


def load_metadata(path: Path) -> dict[str, MetaRow]:
    """Lee el CSV de metadatos públicos. Errores de formato → ValueError con la línea."""
    rows: dict[str, MetaRow] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"spk_id", "nombre_publico", "genero_publico"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"{path}: columnas obligatorias {sorted(required)}, hay {reader.fieldnames}")
        for line, r in enumerate(reader, start=2):
            spk = (r["spk_id"] or "").strip()
            if not spk:
                continue
            if spk in rows:
                raise ValueError(f"{path}:{line}: {spk} repetido")
            try:
                gender = parse_gender(r["genero_publico"] or "")
                role = parse_role(r.get("rol") or "")
            except ValueError as e:
                raise ValueError(f"{path}:{line}: {e}") from None
            rows[spk] = MetaRow(
                public_name=(r["nombre_publico"] or "").strip() or None, gender=gender, role=role
            )
    return rows


# ---------------------------------------------------------------------- voz limpia


def _union(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        elif b > a:
            out.append((a, b))
    return out


def _subtract(base: list[tuple[float, float]], cut: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for a, b in base:
        pieces = [(a, b)]
        for c, d in cut:
            nxt = []
            for x, y in pieces:
                if d <= x or c >= y:
                    nxt.append((x, y))
                    continue
                if c > x:
                    nxt.append((x, c))
                if d < y:
                    nxt.append((d, y))
            pieces = nxt
        out.extend(pieces)
    return [(a, b) for a, b in out if b > a]


def clean_intervals(diar: list[Segment], spk: str, guard_s: float = 0.1) -> list[tuple[float, float]]:
    """Intervalos donde habla ``spk`` y nadie más (con ``guard_s`` de margen a los solapes)."""
    own = _union([(s.start, s.end) for s in diar if s.speaker == spk])
    others = _union([(s.start - guard_s, s.end + guard_s) for s in diar if s.speaker not in (spk, None)])
    return _subtract(own, others)


def windows(
    intervals: list[tuple[float, float]], window_s: float, min_window_s: float, max_windows: int
) -> list[tuple[float, float]]:
    """Trocea en ventanas de ``window_s`` (descarta restos < ``min_window_s``) y, si hay más
    de ``max_windows``, toma una muestra equiespaciada a lo largo del episodio."""
    out = []
    for a, b in intervals:
        t = a
        while b - t >= min_window_s:
            out.append((t, min(b, t + window_s)))
            t += window_s
    if len(out) > max_windows:
        idx = np.linspace(0, len(out) - 1, max_windows).round().astype(int)
        out = [out[i] for i in idx]
    return out


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def speaker_embedding(window_embs: np.ndarray) -> np.ndarray:
    """Media de embeddings normalizados, renormalizada (mismo espacio que en entrenamiento)."""
    return l2norm(l2norm(window_embs).mean(axis=0))


# --------------------------------------------------------------------- clasificador


def load_classifier(path: Path, embedding_model: str) -> dict[str, Any]:
    """Carga el bundle de ``train_gender.py``. Solo modelos propios: joblib ejecuta pickle."""
    import joblib

    if not path.exists():
        raise BackendError(f"No existe {path}: entrena con `python scripts/train_gender.py`")
    bundle = joblib.load(path)
    if bundle.get("embedding_model") != embedding_model:
        raise BackendError(f"{path} se entrenó con {bundle.get('embedding_model')}, no con {embedding_model}")
    return bundle


def predict(bundle: dict[str, Any], emb: np.ndarray) -> tuple[Gender, float]:
    """Devuelve (clase con mayor probabilidad, probabilidad)."""
    model = bundle["model"]
    proba = model.predict_proba(emb.reshape(1, -1))[0]
    k = int(np.argmax(proba))
    return Gender(str(model.classes_[k])), float(proba[k])


def classify(
    wav: Path,
    diar: list[Segment],
    speakers: list[str],
    p: GenderParams,
    embed: Embedder,
    bundle: dict[str, Any],
) -> dict[str, tuple[Gender, float, float]]:
    """``spk -> (género, confianza, voz limpia en s)``. ``U`` si poca voz o poca confianza."""
    out = {}
    for spk in speakers:
        clean = clean_intervals(diar, spk, p.overlap_guard_s)
        clean_s = sum(b - a for a, b in clean)
        if clean_s < p.min_clean_s:
            log.info("gender %s: %.1f s de voz limpia < %.0f s → U", spk, clean_s, p.min_clean_s)
            out[spk] = (Gender.UNKNOWN, 0.0, clean_s)
            continue
        wins = windows(clean, p.window_s, p.min_window_s, p.max_windows)
        clips = [read_window(wav, a, b)[0] for a, b in wins]
        gender, conf = predict(bundle, speaker_embedding(embed(clips)))
        if conf < p.min_conf:
            log.info("gender %s: confianza %.2f < %.2f → U", spk, conf, p.min_conf)
            gender = Gender.UNKNOWN
        out[spk] = (gender, conf, clean_s)
    return out


class EcapaEmbedder:  # pragma: no cover - requiere torch/speechbrain y el modelo
    """ECAPA de SpeechBrain. Compartido por la inferencia y por ``train_gender.py``."""

    def __init__(self, source: str, savedir: str, device: str = "cpu", batch_size: int = 16):
        try:
            import torch
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as e:
            raise BackendError("ECAPA necesita torch y speechbrain: pip install -e '.[gpu]'") from e
        self._torch = torch
        self.batch_size = batch_size
        self.model = EncoderClassifier.from_hparams(
            source=source, savedir=savedir, run_opts={"device": device}
        )
        self.model.eval()

    def __call__(self, clips: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        embs = []
        for i in range(0, len(clips), self.batch_size):
            batch = clips[i : i + self.batch_size]
            n = max(len(c) for c in batch)
            wavs = torch.zeros(len(batch), n)
            for j, c in enumerate(batch):
                wavs[j, : len(c)] = torch.from_numpy(c)
            lens = torch.tensor([len(c) / n for c in batch])
            with torch.no_grad():
                e = self.model.encode_batch(wavs.to(self.model.device), lens.to(self.model.device))
            embs.append(e.squeeze(1).float().cpu().numpy())
        return l2norm(np.concatenate(embs)) if embs else np.zeros((0, 192), dtype=np.float32)


# ------------------------------------------------------------------------------ etapa


def speaking_time(diar: list[Segment]) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for s in diar:
        if s.speaker is not None:
            total[s.speaker] += s.duration
    return dict(total)


def _no_classifier(wav, diar, speakers, p, rt) -> dict[str, tuple[Gender, float, float]]:
    return {s: (Gender.UNKNOWN, 0.0, sum(b - a for a, b in clean_intervals(diar, s, p.overlap_guard_s)))
            for s in speakers}  # fmt: skip


def _ecapa_logreg(wav, diar, speakers, p, rt) -> dict[str, tuple[Gender, float, float]]:
    eligible = [
        s
        for s in speakers
        if sum(b - a for a, b in clean_intervals(diar, s, p.overlap_guard_s)) >= p.min_clean_s
    ]
    if not eligible:  # no cargar modelos si nadie llega a 30 s de voz limpia
        return _no_classifier(wav, diar, speakers, p, rt)
    bundle = load_classifier(Path(p.classifier_path), p.embedding_model)
    device = f"cuda:{p.device_index}" if rt.device == "cuda" else "cpu"
    embedder = EcapaEmbedder(p.embedding_model, p.embedding_savedir, device, p.batch_size)
    return classify(wav, diar, speakers, p, embedder, bundle)


BACKENDS = {"stub": _no_classifier, "ecapa_logreg": _ecapa_logreg}


def build_speakers(
    diar: list[Segment],
    meta: dict[str, MetaRow],
    classified: dict[str, tuple[Gender, float, float]],
    use_public_names: bool,
) -> list[Speaker]:
    """Combina las fuentes por prioridad: metadata > clasificador > unknown."""
    out = []
    for spk, total in sorted(speaking_time(diar).items()):
        row = meta.get(spk)
        name = row.public_name if (row and use_public_names) else None
        clean = classified.get(spk, (None, None, None))[2]
        if row is not None and row.gender is not None:
            conf = 1.0 if row.gender is not Gender.UNKNOWN else 0.0
            sp = Speaker(id=spk, gender=row.gender, gender_source=GenderSource.METADATA, gender_conf=conf)
        elif spk in classified:
            g, conf, _ = classified[spk]
            ran = conf > 0  # el clasificador llegó a ejecutarse
            src = GenderSource.MODEL if ran else GenderSource.NONE
            sp = Speaker(
                id=spk, gender=g if ran else Gender.UNKNOWN, gender_source=src, gender_conf=round(conf, 4)
            )
        else:
            sp = Speaker(id=spk)
        out.append(
            sp.model_copy(
                update={
                    "total_s": round(total, 3),
                    "clean_s": None if clean is None else round(clean, 3),
                    "public_name": name,
                    "role": row.role if row else None,
                }
            )
        )
    return out


def run(
    audio: AudioMeta,
    wav: Path,
    diar: list[Segment],
    cfg: StageCfg,
    rt: RuntimeCfg,
    *,
    annotations_dir: Path = Path("data/annotations"),
    use_public_names: bool = False,
) -> list[Speaker]:
    p = cfg.parsed(GenderParams)
    speakers = sorted(speaking_time(diar))
    csv_path = metadata_path(annotations_dir, audio.episode_id, p)
    meta = load_metadata(csv_path) if csv_path.exists() else {}
    if meta:
        unknown_ids = sorted(set(meta) - set(speakers))
        if unknown_ids:
            log.warning("gender: %s no aparecen en la diarización (¿cambió?): %s", csv_path.name, unknown_ids)
        log.info("gender: metadatos para %d/%d hablantes", len(set(meta) & set(speakers)), len(speakers))
    pending = [s for s in speakers if s not in meta or meta[s].gender is None]
    classified = resolve(BACKENDS, cfg.backend, "gender")(wav, diar, pending, p, rt) if pending else {}
    return build_speakers(diar, meta, classified, use_public_names)


def metadata_path(annotations_dir: Path, episode_id: str, p: GenderParams) -> Path:
    return annotations_dir / p.metadata_csv.format(episode=episode_id)
