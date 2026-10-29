"""Resumen de la ejecución real (Congreso, sesión 202) a partir de los artefactos.

Escribe reports/congreso_202_resultados.json con: presencia sin la Presidencia, tiempo por
rol (pregunta / responde) y género, solapes detectados, eventos, calidad del enlace de voz
y tiempos de proceso. Cada cifra sale de data/interim/<ep>/ y data/annotations/.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from explicometro import io as aio
from explicometro.schemas import Event, Word

man = json.loads(Path("configs/congreso_202.json").read_text(encoding="utf-8"))
tot = defaultdict(float)
words = defaultdict(int)
role_time = defaultdict(list)
ov = {"part_part_n": 0, "part_part_s": 0.0, "chair_part_n": 0, "chair_part_s": 0.0}
events = []
cos_min, margin_min = 1.0, 1.0
proc = defaultdict(float)
audio_s = 0.0
per_ep = []
for b in man["blocks"]:
    ep = b["episode"]
    wd = Path("data/interim") / ep
    rows = {
        r["spk_id"]: r
        for r in csv.DictReader(
            Path(f"data/annotations/speakers_{ep}.csv").read_text(encoding="utf-8").splitlines()
        )
    }
    link = json.loads(Path(f"data/annotations/speakers_{ep}.link.json").read_text(encoding="utf-8"))
    for spk, name in link["assign"].items():
        c = link["cos"][spk]
        cos_min = min(cos_min, c[name])
        others = [v for k, v in c.items() if k != name]
        if others:
            margin_min = min(margin_min, c[name] - max(others))
    segs = aio.read_rttm(wd / "diarization.rttm")
    chair = next(k for k, r in rows.items() if r["rol"].startswith("moderador"))
    for i, s in enumerate(segs):
        for t in segs[i + 1 :]:
            if s.speaker == t.speaker:
                continue
            o = min(s.end, t.end) - max(s.start, t.start)
            if o >= 0.5:
                key = "chair_part" if chair in (s.speaker, t.speaker) else "part_part"
                ov[key + "_n"] += 1
                ov[key + "_s"] += o
    ep_t = defaultdict(float)
    for s in segs:
        if s.speaker != chair:
            ep_t[rows[s.speaker]["genero_publico"]] += s.end - s.start
    for g, v in ep_t.items():
        tot[g] += v
    for w in aio.read_models(wd / "transcript.json", Word):
        if w.speaker and w.speaker != chair:
            words[rows[w.speaker]["genero_publico"]] += 1
    q, r = b["interventions"][0], b["interventions"][1]
    name_t = defaultdict(float)
    for s in segs:
        name_t[rows[s.speaker]["nombre_publico"]] += s.end - s.start
    role_time[("pregunta", q["gender"])].append(name_t[q["name"]])
    role_time[("responde", r["gender"])].append(name_t[r["name"]])
    for e in aio.read_parquet(wd / "events.parquet", Event):
        events.append({"episode": ep, **json.loads(e.model_dump_json())})
    for line in (wd / "timings.jsonl").read_text().splitlines():
        t = json.loads(line)
        proc[t.get("stage", "?")] += float(t.get("seconds", t.get("elapsed_s", 0)) or 0)
    a = json.loads((wd / "audio.json").read_text())
    audio_s += a.get("duration_s", 0)
    per_ep.append(
        {
            "episode": ep,
            "pregunta": f"{q['name']} ({q['gender']})",
            "responde": f"{r['name']} ({r['gender']})",
            "s_pregunta": round(name_t[q["name"]], 1),
            "s_responde": round(name_t[r["name"]], 1),
            "s_presidencia": round(name_t["Presidencia de la Cámara"], 1),
        }
    )
T = tot["M"] + tot["H"]
W = words["M"] + words["H"]
out = {
    "corpus": "Congreso, sesión de control 30/09/2026, 17 preguntas",
    "audio_h": round(audio_s / 3600, 2),
    "participantes_sin_presidencia": {
        "tiempo_M_pct": round(100 * tot["M"] / T, 1),
        "tiempo_H_pct": round(100 * tot["H"] / T, 1),
        "palabras_M_pct": round(100 * words["M"] / W, 1),
        "palabras_H_pct": round(100 * words["H"] / W, 1),
        "tiempo_U_s": round(tot.get("U", 0.0), 1),
    },
    "tiempo_medio_s_por_rol": {
        f"{k[0]}_{k[1]}": {"n": len(v), "media_s": round(sum(v) / len(v), 1)}
        for k, v in sorted(role_time.items())
    },
    "solapes_ge_0_5s": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in ov.items()},
    "eventos": events,
    "enlace_voz": {
        "coseno_min_asignado": round(cos_min, 3),
        "margen_min_frente_a_la_otra_persona": round(margin_min, 3),
    },
    "proceso_s_por_etapa": {k: round(v, 1) for k, v in proc.items()},
    "por_episodio": per_ep,
}
Path("reports").mkdir(exist_ok=True)
Path("reports/congreso_202_resultados.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
)
print(
    json.dumps(
        {k: v for k, v in out.items() if k != "por_episodio"}, ensure_ascii=False, indent=1, default=str
    )
)
