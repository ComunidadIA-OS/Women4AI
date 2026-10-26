"""C3/C4 con juez LLM (cliente falso), modo léxico y C2 por embeddings (embedder falso).

Transcripción sintética (8 turnos) y pares esperados, calculados a mano:

    id  hablante  palabras  nota
    0   00        13        pregunta ("¿...?")
    1   01        25        responde a una pregunta → (0,1) NO se envía
    2   02        25        "te explico ..." → par (1,2)
    3   01        10        B corto → (2,3) NO
    4   02        22        par (3,4)
    5   02        21        mismo hablante que 4 → (4,5) NO
    6   03        1         asentimiento: fuera
    7   01        30        "te equivocas ..." → par (5,7)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import ROOT, S, T
from explicometro import io as aio
from explicometro import llm
from explicometro.config import ReformulationCfg, RuntimeCfg, StageCfg
from explicometro.evaluation import best_threshold, sweep_threshold
from explicometro.events import (
    C2Candidate,
    c2_candidates,
    detect_reformulations,
    has_attribution,
    reformulations_from,
)
from explicometro.schemas import Event, EventType, Gender, Turn

CPU = RuntimeCfg(device="cpu", compute_type="float32")
C3, C4, C2 = EventType.UNSOLICITED_EXPLANATION, EventType.DIRECT_CORRECTION, EventType.REFORMULATION


def words(n: int, base: str = "palabra") -> str:
    return " ".join(f"{base}{i}" for i in range(n))


TXT = {
    0: "¿Qué opinan ustedes sobre la regulación de la inteligencia artificial en Europa hoy?",
    1: words(25, "uno"),
    2: "Mira, te explico cómo funciona la ley porque es muy sencillo " + words(14, "dos"),
    3: words(10, "tres"),
    4: words(22, "cuatro"),
    5: words(21, "cinco"),
    6: "sí",
    7: "No, te equivocas en eso, la norma ya está aprobada desde hace meses " + words(17, "siete"),
}
SPK = {0: "SPK_00", 1: "SPK_01", 2: "SPK_02", 3: "SPK_01", 4: "SPK_02", 5: "SPK_02", 6: "SPK_03", 7: "SPK_01"}
TURNS = [T(i, SPK[i], 10.0 * i, 10.0 * i + 9.0, TXT[i], bc=(i == 6)) for i in range(8)]
SPEAKERS = [
    S("SPK_00", Gender.FEMALE),
    S("SPK_01", Gender.MALE),
    S("SPK_02", Gender.FEMALE),
    S("SPK_03", Gender.MALE),
]


def test_word_counts_match_docstring():
    assert [t.n_words for t in TURNS] == [13, 25, 25, 10, 22, 21, 1, 30]


# ---------------------------------------------------------------------------- pares


def test_select_pairs_by_hand():
    pairs = llm.select_pairs(TURNS, llm.LlmParams())
    assert [(p.a.id, p.b.id) for p in pairs] == [(1, 2), (3, 4), (5, 7)]


@pytest.mark.parametrize(
    ("text", "q"),
    [
        ("¿Qué opinas?", True),
        ("Te pregunto algo. ¿Por qué lo hicisteis? Bueno, adelante.", True),  # penúltima frase
        ("¿Por qué? Lo explico luego. Y además hay más cosas.", False),  # la pregunta queda lejos
        ("Esto es así. Punto.", False),
        ("Pregunta: qué hacemos", False),  # sin signos no se detecta (depende de la puntuación del ASR)
    ],
)
def test_is_question(text, q):
    assert llm.is_question(text) is q


# ------------------------------------------------------------------------ juez falso


class FakeClient:
    """Doble de ``openai.OpenAI``: ``responder(kwargs, n_llamada) -> contenido``."""

    def __init__(self, responder):
        self.responder, self.calls, self.lock = responder, [], threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        with self.lock:
            self.calls.append(kw)
            n = len(self.calls)
        content = self.responder(kw, n)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


def verdict(c3=False, c4=False, ref=False, ev="", conf=0.9):
    return json.dumps(
        {
            "explicacion_no_solicitada": c3,
            "correccion_directa": c4,
            "reformula_idea_de_A": ref,
            "evidencia": ev,
            "confianza": conf,
        }
    )


def smart(kw, n):
    """Juez 'razonable': mira solo RESPUESTA_B."""
    b = kw["messages"][0]["content"].split("RESPUESTA_B:")[1]
    if "te explico" in b:
        return verdict(c3=True, ev="te explico cómo funciona la ley")
    if "te equivocas" in b:
        return verdict(c4=True, ev="No, te equivocas en eso")
    return verdict()


def run_llm(tmp_path, responder, **params):
    client = FakeClient(responder)
    p = {"cache_path": str(tmp_path / "cache.sqlite"), **params}
    ev = llm.run(
        TURNS,
        SPEAKERS,
        StageCfg(backend="openai_compatible", params=p),
        CPU,
        workdir=tmp_path / "ep",
        client_factory=lambda s, pp: client,
    )
    return ev, client


def test_llm_events_and_blinding(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "modelo-de-test")
    named = [s.model_copy(update={"public_name": f"Nombre{i}"}) for i, s in enumerate(SPEAKERS)]
    client = FakeClient(smart)
    ev = llm.run(
        TURNS,
        named,
        StageCfg(backend="openai_compatible", params={"cache_path": None}),
        CPU,
        client_factory=lambda s, p: client,
    )
    assert [(e.type, e.src_spk, e.dst_spk, e.evidence) for e in ev] == [
        (C3, "SPK_02", "SPK_01", "te explico cómo funciona la ley"),
        (C4, "SPK_01", "SPK_02", "No, te equivocas en eso"),
    ]
    template = llm.load_prompt("judge_es.txt")
    sent = sorted(c["messages"][0]["content"] for c in client.calls)
    expected = sorted(
        llm.render_prompt(template, *llm.pair_texts(p, llm.LlmParams()))
        for p in llm.select_pairs(TURNS, llm.LlmParams())
    )
    assert sent == expected  # el prompt es exactamente plantilla + textos: nada más
    for c in client.calls:
        content = c["messages"][0]["content"]
        assert "SPK_" not in content and "Nombre" not in content
        assert c["temperature"] == 0.0 and c["model"] == "modelo-de-test" and len(c["messages"]) == 1


def test_retry_once_then_valid(tmp_path):
    def flaky(kw, n):
        return (
            "Claro, aquí tienes: {no es json"
            if len(kw["messages"]) == 1
            else smart({"messages": kw["messages"][:1]}, n)
        )

    ev, client = run_llm(tmp_path, flaky)
    assert len(ev) == 2 and len(client.calls) == 6  # 3 pares × 2 intentos
    second = client.calls[1]["messages"]
    assert second[1]["role"] == "assistant" and "no válida" in second[2]["content"]
    usage = json.loads((tmp_path / "ep" / "llm_usage.json").read_text())
    assert (usage["calls"], usage["retries"], usage["failed"]) == (6, 3, 0)


@pytest.mark.parametrize(
    "bad",
    [
        verdict(c4=True, ev="una frase que B nunca dijo"),  # evidencia no literal
        verdict(c4=True, ev=words(21)),  # > 20 palabras
        verdict(c4=True, ev=""),  # etiqueta sin evidencia
        verdict().replace("false", '"false"', 1),  # tipo incorrecto
        verdict(conf=1.5),
    ],
)
def test_invalid_twice_is_failure_without_events(tmp_path, bad):
    ev, client = run_llm(tmp_path, lambda kw, n: bad)
    assert ev == [] and len(client.calls) == 6
    lines = (tmp_path / "ep" / "llm_judgments.jsonl").read_text().splitlines()
    assert all(not json.loads(x)["ok"] for x in lines)


def test_json_fences_and_extra_text_are_tolerated():
    out = llm.parse_judgement(
        "```json\n" + verdict(c3=True, ev="te explico") + "\n```", "Bueno, te explico algo"
    )
    assert out.explicacion_no_solicitada and out.evidencia == "te explico"


def test_min_confidence_filters(tmp_path):
    ev, _ = run_llm(tmp_path, lambda kw, n: smart(kw, n).replace("0.9", "0.4"), min_confidence=0.5)
    assert ev == []


def test_cache_hits_and_usage_cost(tmp_path):
    prices = {"price_input_per_mtok": 1.0, "price_output_per_mtok": 2.0}
    ev1, _ = run_llm(tmp_path, smart, **prices)
    u1 = json.loads((tmp_path / "ep" / "llm_usage.json").read_text())
    assert (u1["pairs"], u1["cached"], u1["calls"]) == (3, 0, 3)
    assert (u1["prompt_tokens"], u1["completion_tokens"]) == (300, 60)
    assert u1["cost_eur"] == pytest.approx(3 * (100 * 1.0 + 20 * 2.0) / 1e6)
    assert u1["latency_mean_s"] is not None
    ev2, c2 = run_llm(tmp_path, smart, **prices)
    u2 = json.loads((tmp_path / "ep" / "llm_usage.json").read_text())
    assert ev2 == ev1 and c2.calls == [] and (u2["cached"], u2["calls"], u2["cost_eur"]) == (3, 0, 0.0)


def test_cache_key_depends_on_model_prompt_and_texts():
    k = llm.cache_key("m", "p1", 0.0, "a", "b")
    assert k != llm.cache_key("m2", "p1", 0.0, "a", "b") != llm.cache_key("m", "p2", 0.0, "a", "b")
    assert k != llm.cache_key("m", "p1", 0.0, "a", "b2") and k == llm.cache_key("m", "p1", 0.0, "a", "b")


def test_json_mode_fallback_when_server_rejects(tmp_path):
    def picky(kw, n):
        if "response_format" in kw:
            return RuntimeError("400: response_format json_object no soportado")
        return smart(kw, n)

    ev, client = run_llm(tmp_path, picky)
    assert len(ev) == 2 and "response_format" in client.calls[0]
    assert all("response_format" not in c for c in client.calls[1:])


def test_long_turns_are_clipped_end_of_a_start_of_b():
    a = T(0, "SPK_00", 0, 1, words(300, "a"))
    b = T(1, "SPK_01", 1, 2, words(300, "b"))
    ta, tb = llm.pair_texts(llm.Pair(a, b), llm.LlmParams(max_words_a=10, max_words_b=10))
    assert ta == "… " + " ".join(f"a{i}" for i in range(290, 300))
    assert tb == " ".join(f"b{i}" for i in range(10)) + " …"


# ---------------------------------------------------------------------------- léxico


def test_lexicon_only_mode(tmp_path):
    ev = llm.run(TURNS, SPEAKERS, StageCfg(backend="lexicon"), CPU, workdir=tmp_path)
    assert [(e.type, e.src_spk) for e in ev] == [(C3, "SPK_02"), (C4, "SPK_01")]
    assert "te explico" in ev[0].evidence and "te equivocas" in ev[1].evidence
    assert json.loads((tmp_path / "llm_usage.json").read_text())["backend"] == "lexicon"


# -------------------------------------------------------------------------------- C2


def bow_embed(texts: list[str]) -> np.ndarray:
    """Bolsa de palabras normalizada: coseno = solapamiento léxico (calculable a mano)."""
    vocab = sorted({w for t in texts for w in t.lower().replace(",", "").split()})
    idx = {w: i for i, w in enumerate(vocab)}
    E = np.zeros((len(texts), len(vocab)))
    for r, t in enumerate(texts):
        for w in t.lower().replace(",", "").split():
            E[r, idx[w]] += 1
    return E / np.linalg.norm(E, axis=1, keepdims=True)


IDEA = (
    "la transición energética exige invertir en redes eléctricas y almacenamiento "
    "para llegar a todos los hogares"
)


def test_c2_candidates_and_attribution():
    assert len(IDEA.split()) == 16
    turns = [
        T(0, "SPK_00", 0, 10, IDEA),
        T(1, "SPK_02", 10, 20, words(16, "futbol")),
        T(2, "SPK_00", 20, 22, "corto y sin más"),  # < 15 palabras: fuera
        T(
            3,
            "SPK_01",
            22,
            32,
            " ".join(reversed(IDEA.split())),  # mismas palabras, otro orden
        ),
        T(4, "SPK_02", 32, 42, "como decía Marta, " + IDEA),  # atribuye: no es C2
    ]
    cfg = ReformulationCfg(cos_threshold=0.9)
    cands = c2_candidates(turns, bow_embed, cfg)
    got = {(c.b.id, c.a.id): round(c.cos, 6) for c in cands}
    assert got == {(1, 0): 0.0, (3, 1): 0.0, (3, 0): 1.0}  # el 4 se descarta por atribución
    ev = detect_reformulations(turns, bow_embed, cfg)
    assert [(e.type, e.src_spk, e.dst_spk, e.score) for e in ev] == [(C2, "SPK_01", "SPK_00", 1.0)]
    assert has_attribution("Tiene razón Pere, però...", cfg.attribution_patterns)
    assert has_attribution("com deia la Núria", cfg.attribution_patterns)
    assert not has_attribution("yo creo que la red es clave", cfg.attribution_patterns)


def test_c2_window_is_five_previous_turns_of_others():
    others = [
        T(i + 1, f"SPK_0{1 + i % 2}", 10 * (i + 1), 10 * (i + 1) + 9, words(16, f"x{i}")) for i in range(5)
    ]
    turns = [T(0, "SPK_00", 0, 9, IDEA), *others, T(6, "SPK_03", 60, 69, IDEA)]
    cands = c2_candidates(turns, bow_embed, ReformulationCfg())
    assert {c.a.id for c in cands if c.b.id == 6} == {1, 2, 3, 4, 5}  # el turno 0 queda fuera de la ventana


def test_c2_without_tau_emits_nothing_and_warns(caplog):
    turns = [T(0, "SPK_00", 0, 10, IDEA), T(1, "SPK_01", 10, 20, IDEA)]
    assert detect_reformulations(turns, bow_embed, ReformulationCfg()) == []
    assert "sin calibrar" in caplog.text


# ------------------------------------------------------------------------ calibración


def cand(b_id: int, cos: float, t: float) -> C2Candidate:
    return C2Candidate(b=T(b_id, "SPK_01", t, t + 5, words(16)), a=T(99, "SPK_00", 0, 1, words(16)), cos=cos)


def test_sweep_threshold_picks_best_f1_highest_tau():
    cands = [cand(1, 0.95, 10), cand(2, 0.85, 20), cand(3, 0.80, 30), cand(4, 0.60, 40)]
    gold = [Event(type=C2, src_spk="SPK_01", dst_spk="SPK_00", t_start=t, t_end=t + 5) for t in (10, 20)]
    taus = [round(x, 2) for x in np.arange(0.5, 1.0, 0.01)]
    curve = sweep_threshold([(lambda tau: reformulations_from(cands, tau), gold)], taus, C2)
    by = {c["tau"]: c for c in curve}
    assert (by[0.8]["tp"], by[0.8]["fp"]) == (2, 1) and by[0.9]["fn"] == 1
    assert best_threshold(curve)["tau"] == 0.85  # F1 = 1 en (0,80; 0,85]; el más alto


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _gold_setup(tmp_path: Path, gold: list[Event], turns: list[Turn]) -> Path:
    aio.write_models(tmp_path / "ann" / "ep1" / "events.json", gold)
    aio.write_parquet(tmp_path / "interim" / "ep1" / "turns.parquet", turns, Turn)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"paths: {{annotations: {tmp_path / 'ann'}, interim: {tmp_path / 'interim'}, "
        f"processed: {tmp_path / 'out'}}}\n"
        f"stages: {{llm: {{backend: openai_compatible, params: {{cache_path: null}}}}}}\n",
        encoding="utf-8",
    )
    return cfg


def test_calibrate_c2_script(tmp_path):
    turns = [
        T(0, "SPK_00", 0, 10, IDEA),
        T(1, "SPK_01", 10, 20, IDEA + " sin duda"),
        T(2, "SPK_00", 20, 30, words(16, "otra")),
        T(3, "SPK_01", 30, 40, words(16, "otra") + " y algo"),
    ]
    gold = [Event(type=C2, src_spk="SPK_01", dst_spk="SPK_00", t_start=10, t_end=20)]
    cfg = _gold_setup(tmp_path, gold, turns)
    assert _load("calibrate_c2").main(["--config", str(cfg)], embedder_factory=lambda: bow_embed) == 0
    rep = json.loads((tmp_path / "out" / "c2_calibration.json").read_text())
    # cos(1,0) = 16/√16·√18 = 0,9428 (verdadero) · cos(3,2) = 16/√16·√18 = 0,9428 (falso) → empate: P = 0,5
    assert rep["best"]["tp"] == 1 and rep["n_gold"] == 1


def test_ablation_script_lexicon_vs_llm(tmp_path):
    gold = [
        Event(type=C3, src_spk="SPK_02", dst_spk="SPK_01", t_start=20, t_end=29),
        Event(type=C4, src_spk="SPK_01", dst_spk="SPK_02", t_start=70, t_end=79),
    ]
    cfg = _gold_setup(tmp_path, gold, TURNS)
    client = FakeClient(lambda kw, n: verdict())  # LLM que nunca marca nada
    assert _load("ablation_llm").main(["--config", str(cfg)], client_factory=lambda s, p: client) == 0
    rep = json.loads((tmp_path / "out" / "ablation_llm.json").read_text())["results"]
    assert rep["lexicon"]["C3_unsolicited_explanation"]["tp"] == 1
    assert rep["lexicon"]["C4_direct_correction"]["f1"] == 1.0
    assert rep["llm"]["C4_direct_correction"]["fn"] == 1


def test_real_openai_sdk_against_local_fake_server(tmp_path, monkeypatch):
    """El cliente ``openai`` real contra un servidor OpenAI-compatible falso en localhost."""
    import http.server

    seen: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
            content = smart(body, len(seen))
            payload = json.dumps(
                {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("LLM_API_KEY", "clave-de-test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    try:
        ev = llm.run(
            TURNS,
            SPEAKERS,
            StageCfg(backend="openai_compatible", params={"cache_path": None}),
            CPU,
            workdir=tmp_path,
        )
    finally:
        srv.shutdown()
    assert [e.type for e in ev] == [C3, C4]
    assert len(seen) == 3 and all(s["path"] == "/v1/chat/completions" for s in seen)
    assert seen[0]["auth"] == "Bearer clave-de-test" and seen[0]["temperature"] == 0.0
    assert seen[0]["response_format"] == {"type": "json_object"}
    usage = json.loads((tmp_path / "llm_usage.json").read_text())
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (33, 21)
