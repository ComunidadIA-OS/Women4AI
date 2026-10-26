"""C3 (explicación no solicitada) y C4 (corrección directa) con un LLM abierto como juez.

Rigor:
- **Cegado**: el prompt solo lleva el texto de A y de B. Ni IDs de hablante, ni género, ni
  nombres de metadatos: la asimetría no la puede introducir el juez (ver docs/llm.md para
  el límite: la gramática del propio texto puede revelar género).
- Solo se juzgan pares (A, B) con A ≠ B, B ≥ 20 palabras y A que no es pregunta.
- Salida JSON validada con pydantic; la ``evidencia`` debe ser cita literal de B (≤ 20
  palabras). Un reintento si falla. Temperatura 0. Caché por hash de (modelo, prompt, A, B).
- Coste, tokens y latencia en ``llm_usage.json``; cada juicio en ``llm_judgments.jsonl``.
- Backend ``lexicon``: mismas reglas de pares, etiquetas por léxico (ablación sin LLM).

Cliente OpenAI-compatible: ``LLM_BASE_URL``, ``LLM_API_KEY`` y ``LLM_MODEL`` (.env) tienen
prioridad sobre ``stages.llm.params``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import sqlite3
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

from .backends import BackendError, resolve
from .config import Params, RuntimeCfg, StageCfg, env
from .schemas import Event, EventType, Speaker, Turn

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent / "prompts"

DEFAULT_LEXICON_C3 = [
    r"\bte (lo )?explico\b",
    r"\bd[ée]jame que te (lo )?explique\b",
    r"\blo que (tienes|tiene) que entender\b",
    r"\bpara que (lo )?entiendas\b",
    r"\bes muy (sencillo|simple)\b",
    r"\bb[áa]sicamente\b",
    r"\ba ver, (mira|es que)\b",
    r"\bdeixa'?m que t'?expliqui\b",
    r"\bel que has d'entendre\b",
]
DEFAULT_LEXICON_C4 = [
    r"\bte equivocas\b",
    r"\bse equivoca\b",
    r"\best[áa]s? equivocad[oa]\b",
    r"\beso (no es|es) (así|verdad|cierto|falso)\b",
    r"\bno tienes? raz[óo]n\b",
    r"\bno,? no,? no\b",
    r"\ben realidad no\b",
    r"\bperdona,? pero no\b",
    r"\bt'?equivoques\b",
    r"\bs'?equivoca\b",
    r"\bno [ée]s (així|veritat|cert)\b",
]


class LlmParams(Params):
    model: str = "BSC-LT/salamandra-7b-instruct-2606"
    base_url: str | None = None
    prompt: str = "judge_es.txt"
    temperature: float = Field(default=0.0, ge=0)
    max_tokens: int = Field(default=256, ge=16)
    json_mode: bool = True
    timeout_s: float = Field(default=60.0, gt=0)
    http_retries: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=4, ge=1)
    min_b_words: int = Field(default=20, ge=1)
    min_a_words: int = Field(default=3, ge=1)
    max_words_a: int = Field(default=250, ge=10)
    max_words_b: int = Field(default=250, ge=10)
    min_confidence: float = Field(default=0.5, ge=0, le=1)
    cache_path: str | None = "data/interim/_llm_cache.sqlite"
    price_input_per_mtok: float = Field(default=0.0, ge=0)
    price_output_per_mtok: float = Field(default=0.0, ge=0)
    lexicon_c3: list[str] = Field(default_factory=lambda: list(DEFAULT_LEXICON_C3))
    lexicon_c4: list[str] = Field(default_factory=lambda: list(DEFAULT_LEXICON_C4))


def llm_settings(p: LlmParams) -> dict[str, str | None]:
    """Endpoint y modelo efectivos (.env > YAML). Nunca incluye la clave."""
    return {"base_url": env("LLM_BASE_URL", p.base_url), "model": env("LLM_MODEL", p.model)}


# ------------------------------------------------------------------------------ pares


@dataclass(frozen=True)
class Pair:
    a: Turn
    b: Turn


_SENT = re.compile(r"(?<=[.!?…])\s+")


def is_question(text: str) -> bool:
    """A 'era pregunta' si alguna de sus dos últimas frases contiene '?' (o abre con '¿')."""
    sentences = [s for s in _SENT.split(text.strip()) if s]
    return any("?" in s or s.lstrip().startswith("¿") for s in sentences[-2:])


def select_pairs(turns: list[Turn], p: LlmParams) -> list[Pair]:
    """(A, B) = turno inmediatamente anterior a B (sin asentimientos), de otra persona."""
    floor = sorted((t for t in turns if not t.is_backchannel), key=lambda t: (t.start, t.end))
    pairs = []
    for a, b in itertools.pairwise(floor):
        if a.speaker == b.speaker or b.n_words < p.min_b_words or a.n_words < p.min_a_words:
            continue
        if is_question(a.text):
            continue
        pairs.append(Pair(a, b))
    return pairs


# ------------------------------------------------------------------------------ prompt


def _clip(text: str, max_words: int, keep: str) -> str:
    words = " ".join(text.split()).replace('"', "'").split()
    if len(words) <= max_words:
        return " ".join(words)
    return "… " + " ".join(words[-max_words:]) if keep == "end" else " ".join(words[:max_words]) + " …"


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def render_prompt(template: str, a_text: str, b_text: str) -> str:
    """Solo se sustituyen ``{a}`` y ``{b}`` (el JSON de ejemplo lleva llaves). Cegado: no se
    pasa nada más que el texto."""
    return template.replace("{a}", a_text).replace("{b}", b_text)


def pair_texts(pair: Pair, p: LlmParams) -> tuple[str, str]:
    """Final de A (lo que B responde) y principio de B."""
    return _clip(pair.a.text, p.max_words_a, "end"), _clip(pair.b.text, p.max_words_b, "start")


# ------------------------------------------------------------------------------ salida


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    explicacion_no_solicitada: StrictBool
    correccion_directa: StrictBool
    reformula_idea_de_A: StrictBool
    evidencia: str = ""
    confianza: float = Field(ge=0, le=1)

    @field_validator("evidencia")
    @classmethod
    def _max_20_words(cls, v: str) -> str:
        v = v.strip().strip('"«»“”').strip()
        if len(v.split()) > 20:
            raise ValueError("evidencia de más de 20 palabras")
        return v

    @property
    def any_label(self) -> bool:
        return self.explicacion_no_solicitada or self.correccion_directa or self.reformula_idea_de_A


class JudgeError(ValueError):
    pass


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def extract_json(raw: str) -> dict[str, Any]:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise JudgeError("no hay un objeto JSON en la respuesta")
    try:
        return json.loads(s[i : j + 1])
    except json.JSONDecodeError as e:
        raise JudgeError(f"JSON inválido: {e.msg}") from None


def parse_judgement(raw: str, b_text: str) -> JudgeOutput:
    """Valida tipos y que la evidencia sea cita literal de B (si hay alguna etiqueta)."""
    try:
        out = JudgeOutput.model_validate(extract_json(raw))
    except ValidationError as e:
        raise JudgeError(
            "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        ) from None
    if out.any_label:
        if not out.evidencia:
            raise JudgeError("hay etiquetas positivas pero la evidencia está vacía")
        if _norm(out.evidencia) not in _norm(b_text):
            raise JudgeError("la evidencia no es una cita literal de RESPUESTA_B")
    return out


# ------------------------------------------------------------------------- caché y uso


def cache_key(model: str, prompt_sha: str, temperature: float, a: str, b: str) -> str:
    blob = json.dumps([model, prompt_sha, temperature, a, b], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class JudgeCache:
    """Caché SQLite (se guarda en el hilo principal). Solo respuestas del modelo, no errores de red."""

    def __init__(self, path: Path | None):
        self.conn = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(path))
            self.conn.execute("CREATE TABLE IF NOT EXISTS judge (key TEXT PRIMARY KEY, record TEXT NOT NULL)")

    def get(self, key: str) -> dict[str, Any] | None:
        if self.conn is None:
            return None
        row = self.conn.execute("SELECT record FROM judge WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        if self.conn is not None:
            self.conn.execute(
                "INSERT OR REPLACE INTO judge VALUES (?, ?)", (key, json.dumps(record, ensure_ascii=False))
            )
            self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()


@dataclass
class Usage:
    backend: str = ""
    model: str | None = None
    prompt_sha: str | None = None
    pairs: int = 0
    cached: int = 0
    calls: int = 0
    retries: int = 0
    failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: list[float] = field(default_factory=list)
    cost_eur: float = 0.0

    def summary(self) -> dict[str, Any]:
        lat = sorted(self.latency_s)
        d = asdict(self)
        d.pop("latency_s")
        d |= {
            "latency_total_s": round(sum(lat), 3),
            "latency_mean_s": round(sum(lat) / len(lat), 3) if lat else None,
            "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3) if lat else None,
            "cost_eur": round(self.cost_eur, 6),
        }
        return d


# ------------------------------------------------------------------------- juez (LLM)

Client = Any  # openai.OpenAI o un doble de test con .chat.completions.create(...)


def make_client(settings: dict[str, str | None], p: LlmParams) -> Client:  # pragma: no cover - red
    from openai import OpenAI

    if not settings["base_url"]:
        raise BackendError("Falta LLM_BASE_URL en .env (vLLM local u OVH AI Endpoints)")
    return OpenAI(
        base_url=settings["base_url"],
        api_key=env("LLM_API_KEY", "EMPTY"),
        timeout=p.timeout_s,
        max_retries=p.http_retries,
    )


def _complete(
    client: Client, model: str, messages: list[dict], p: LlmParams, json_mode: bool
) -> tuple[str, Any, float]:
    kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": p.temperature,
        "max_tokens": p.max_tokens,
    }
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kw)
    return resp.choices[0].message.content or "", getattr(resp, "usage", None), time.perf_counter() - t0


def judge(
    client: Client, model: str, prompt: str, b_text: str, p: LlmParams, json_mode: bool = True
) -> dict[str, Any]:
    """Un par: hasta 2 intentos (el 2º le devuelve el error al modelo). Devuelve el registro."""
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    rec: dict[str, Any] = {
        "ok": False,
        "output": None,
        "error": None,
        "attempts": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_s": 0.0,
    }
    for attempt in (1, 2):
        raw, usage, dt = _complete(client, model, messages, p, json_mode)
        rec["attempts"] = attempt
        rec["latency_s"] += dt
        rec["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        rec["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        rec["raw"] = raw
        try:
            rec["output"] = parse_judgement(raw, b_text).model_dump()
            rec["ok"], rec["error"] = True, None
            return rec
        except JudgeError as e:
            rec["error"] = str(e)
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"Respuesta no válida ({e}). Responde SOLO con el JSON pedido; "
                    "la evidencia debe copiarse literalmente de RESPUESTA_B.",
                },
            ]
    return rec


def judge_pairs(
    pairs: list[Pair], p: LlmParams, client: Client, model: str, cache: JudgeCache, usage: Usage
) -> list[dict[str, Any]]:
    """Juzga los pares (caché primero, el resto en paralelo). Devuelve un registro por par."""
    template = load_prompt(p.prompt)
    prompt_sha = hashlib.sha256(template.encode()).hexdigest()[:16]
    usage.model, usage.prompt_sha, usage.pairs = model, prompt_sha, len(pairs)
    jobs: list[tuple[int, str, str, str]] = []
    records: list[dict[str, Any] | None] = [None] * len(pairs)
    for i, pair in enumerate(pairs):
        a, b = pair_texts(pair, p)
        key = cache_key(model, prompt_sha, p.temperature, a, b)
        hit = cache.get(key)
        if hit is not None:
            records[i] = hit | {"cached": True}
            usage.cached += 1
        else:
            jobs.append((i, key, render_prompt(template, a, b), b))

    json_mode = p.json_mode

    def run(job: tuple[int, str, str, str]) -> tuple[int, str, dict[str, Any]]:
        i, key, prompt, b = job
        return i, key, judge(client, model, prompt, b, p, json_mode)

    if jobs:
        try:  # el primero en serie: si el servidor no admite response_format, se desactiva
            first = [run(jobs[0])]
        except Exception as e:  # depende del servidor
            if not json_mode or "response_format" not in str(e):
                raise
            log.warning("llm: el servidor no admite response_format=json_object; sigo sin él")
            json_mode = False
            first = [run(jobs[0])]
        with ThreadPoolExecutor(max_workers=p.max_concurrency) as pool:
            done = first + list(pool.map(run, jobs[1:]))
        for i, key, rec in done:
            cache.put(key, rec)
            records[i] = rec | {"cached": False}
            usage.calls += rec["attempts"]
            usage.retries += rec["attempts"] - 1
            usage.prompt_tokens += rec["prompt_tokens"]
            usage.completion_tokens += rec["completion_tokens"]
            usage.latency_s.append(rec["latency_s"])
            usage.cost_eur += (
                rec["prompt_tokens"] * p.price_input_per_mtok
                + rec["completion_tokens"] * p.price_output_per_mtok
            ) / 1e6
    usage.failed = sum(1 for r in records if r is not None and not r["ok"])
    return [r for r in records if r is not None]


def events_from_judgements(pairs: list[Pair], records: list[dict[str, Any]], min_conf: float) -> list[Event]:
    """C3 y C4 (B → A) con confianza >= ``min_conf``. ``reformula_idea_de_A`` no genera
    evento (C2 es por embeddings); queda en ``llm_judgments.jsonl`` para la ablación."""
    events = []
    for pair, rec in zip(pairs, records, strict=True):
        out = rec.get("output")
        if not rec.get("ok") or out is None or out["confianza"] < min_conf:
            continue
        for flag, etype in (
            ("explicacion_no_solicitada", EventType.UNSOLICITED_EXPLANATION),
            ("correccion_directa", EventType.DIRECT_CORRECTION),
        ):
            if out[flag]:
                events.append(
                    Event(
                        type=etype,
                        src_spk=pair.b.speaker,
                        dst_spk=pair.a.speaker,
                        t_start=pair.b.start,
                        t_end=pair.b.end,
                        evidence=out["evidencia"],
                        score=out["confianza"],
                    )
                )
    return events


# --------------------------------------------------------------------------- léxico


def lexicon_judgement(b_text: str, p: LlmParams) -> dict[str, Any]:
    """Etiquetas por léxico (sin LLM): primera coincidencia de C3/C4 en B como evidencia."""

    def first(patterns: list[str]) -> str | None:
        for pat in patterns:
            m = re.search(pat, b_text, flags=re.IGNORECASE)
            if m:
                words = b_text[m.start() :].split()
                return " ".join(words[: min(20, max(1, len(m.group(0).split())))])
        return None

    c3, c4 = first(p.lexicon_c3), first(p.lexicon_c4)
    out = {
        "explicacion_no_solicitada": c3 is not None,
        "correccion_directa": c4 is not None,
        "reformula_idea_de_A": False,
        "evidencia": c4 or c3 or "",
        "confianza": 1.0,
    }
    return {"ok": True, "output": out, "error": None, "attempts": 0, "cached": False}


# ----------------------------------------------------------------------------- etapa


def _write_side_files(workdir: Path | None, pairs: list[Pair], records: list[dict], usage: Usage) -> None:
    if workdir is None:
        return
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / "llm_judgments.jsonl").open("w", encoding="utf-8") as f:
        for pair, rec in zip(pairs, records, strict=True):
            row = {
                "a_turn": pair.a.id,
                "b_turn": pair.b.id,
                "src_spk": pair.b.speaker,
                "dst_spk": pair.a.speaker,
                **{k: rec.get(k) for k in ("ok", "output", "error", "attempts", "cached", "latency_s")},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (workdir / "llm_usage.json").write_text(json.dumps(usage.summary(), ensure_ascii=False, indent=2) + "\n")
    s = usage.summary()
    log.info(
        "llm[%s]: %d pares, %d de caché, %d llamadas (%d reintentos), %d fallidos, %d+%d tokens, "
        "%.1f s, %.4f €",
        s["backend"],
        s["pairs"],
        s["cached"],
        s["calls"],
        s["retries"],
        s["failed"],
        s["prompt_tokens"],
        s["completion_tokens"],
        s["latency_total_s"],
        s["cost_eur"],
    )


def _stub(
    turns: list[Turn], p: LlmParams, rt: RuntimeCfg, workdir: Path | None, client_factory: Callable | None
) -> list[Event]:
    return []


def _lexicon(
    turns: list[Turn], p: LlmParams, rt: RuntimeCfg, workdir: Path | None, client_factory: Callable | None
) -> list[Event]:
    pairs = select_pairs(turns, p)
    records = [lexicon_judgement(pair_texts(pr, p)[1], p) for pr in pairs]
    _write_side_files(workdir, pairs, records, Usage(backend="lexicon", pairs=len(pairs)))
    return events_from_judgements(pairs, records, 0.0)


def _openai_compatible(
    turns: list[Turn], p: LlmParams, rt: RuntimeCfg, workdir: Path | None, client_factory: Callable | None
) -> list[Event]:
    pairs = select_pairs(turns, p)
    usage = Usage(backend="openai_compatible")
    if not pairs:
        _write_side_files(workdir, [], [], usage)
        return []
    settings = llm_settings(p)
    client = (client_factory or make_client)(settings, p)
    cache = JudgeCache(Path(p.cache_path) if p.cache_path else None)
    try:
        records = judge_pairs(pairs, p, client, settings["model"] or p.model, cache, usage)
    finally:
        cache.close()
    _write_side_files(workdir, pairs, records, usage)
    return events_from_judgements(pairs, records, p.min_confidence)


BACKENDS = {"stub": _stub, "lexicon": _lexicon, "openai_compatible": _openai_compatible}


def run(
    turns: list[Turn],
    speakers: list[Speaker],
    cfg: StageCfg,
    rt: RuntimeCfg,
    *,
    workdir: Path | None = None,
    client_factory: Callable | None = None,
) -> list[Event]:
    """``speakers`` se recibe por contrato pero **no se usa**: el juez está cegado al género."""
    del speakers
    p = cfg.parsed(LlmParams)
    return resolve(BACKENDS, cfg.backend, "llm")(turns, p, rt, workdir, client_factory)
