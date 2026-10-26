"""Ablación de C3/C4: juez LLM frente a modo solo-léxico, sobre el gold set.

    python scripts/ablation_llm.py              # léxico + LLM (usa la caché del juez)
    python scripts/ablation_llm.py --no-llm     # solo léxico

Para cada episodio con ``data/annotations/<ep>/events.json`` y ``turns.parquet``, aplica los
mismos pares (A, B) con los dos modos y puntúa C3 y C4 contra el gold (P/R/F1 micro). También
puntúa la etiqueta ``reformula_idea_de_A`` del LLM contra el C2 del gold, para compararla con
los embeddings. Resultado en ``data/processed/ablation_llm.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from explicometro import io as aio
from explicometro import llm
from explicometro.config import load_config
from explicometro.evaluation import PRF, score_events
from explicometro.report import fmt3
from explicometro.schemas import Event, EventType, Turn

TYPES = [EventType.UNSOLICITED_EXPLANATION, EventType.DIRECT_CORRECTION, EventType.REFORMULATION]


def reformulation_events(pairs: list[llm.Pair], records: list[dict]) -> list[Event]:
    return [
        Event(
            type=EventType.REFORMULATION,
            src_spk=pr.b.speaker,
            dst_spk=pr.a.speaker,
            t_start=pr.b.start,
            t_end=pr.b.end,
            evidence=r["output"]["evidencia"],
            score=r["output"]["confianza"],
        )
        for pr, r in zip(pairs, records, strict=True)
        if r.get("ok") and r["output"]["reformula_idea_de_A"]
    ]


def main(argv: list[str] | None = None, client_factory: Callable | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config if a.config.exists() else None)
    p = cfg.stages.llm.parsed(llm.LlmParams)
    modes = ["lexicon"] if a.no_llm else ["lexicon", "llm"]
    totals = {m: {t: PRF() for t in TYPES} for m in modes}
    usage = llm.Usage(backend="openai_compatible")
    client = cache = None
    n_eps = 0
    for gold_path in sorted(cfg.paths.annotations.glob("*/events.json")):
        ep = gold_path.parent.name
        turns_path = cfg.paths.interim / ep / "turns.parquet"
        if not turns_path.exists():
            print(f"[{ep}] sin turns.parquet: se omite")
            continue
        n_eps += 1
        gold = aio.read_models(gold_path, Event)
        pairs = llm.select_pairs(aio.read_parquet(turns_path, Turn), p)
        for mode in modes:
            if mode == "lexicon":
                recs = [llm.lexicon_judgement(llm.pair_texts(pr, p)[1], p) for pr in pairs]
                pred = llm.events_from_judgements(pairs, recs, 0.0)
            else:
                if client is None:
                    settings = llm.llm_settings(p)
                    client = (client_factory or llm.make_client)(settings, p)
                    cache = llm.JudgeCache(Path(p.cache_path) if p.cache_path else None)
                recs = llm.judge_pairs(
                    pairs, p, client, llm.llm_settings(p)["model"] or p.model, cache, usage
                )
                pred = llm.events_from_judgements(pairs, recs, p.min_confidence) + reformulation_events(
                    pairs, recs
                )
            scores = score_events(pred, [g for g in gold if g.type in TYPES], a.tol)
            for t in TYPES:
                s, tot = scores.get(t, PRF()), totals[mode][t]
                totals[mode][t] = PRF(tp=tot.tp + s.tp, fp=tot.fp + s.fp, fn=tot.fn + s.fn)
    if cache is not None:
        cache.close()
    if n_eps == 0:
        print("No hay episodios anotados con turns.parquet.")
        return 1

    f = fmt3
    print(f"{n_eps} episodios · tolerancia {a.tol} s")
    print(f"{'modo':8} {'tipo':28} {'P':>6} {'R':>6} {'F1':>6}  tp/fp/fn")
    report: dict = {"n_episodes": n_eps, "tol_s": a.tol, "results": {}}
    for m in modes:
        for t in TYPES:
            s = totals[m][t]
            if m == "lexicon" and t == EventType.REFORMULATION:
                continue  # el léxico no etiqueta C2
            print(f"{m:8} {t.value:28} {f(s.precision)} {f(s.recall)} {f(s.f1)}  {s.tp}/{s.fp}/{s.fn}")
            report["results"].setdefault(m, {})[t.value] = s.model_dump() | {"f1": s.f1}
    if "llm" in modes:
        report["llm_usage"] = usage.summary()
        u = report["llm_usage"]
        print(f"LLM: {u['pairs']} pares, {u['cached']} de caché, {u['latency_total_s']} s, {u['cost_eur']} €")
    out = cfg.paths.processed / "ablation_llm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
