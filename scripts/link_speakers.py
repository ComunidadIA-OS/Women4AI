"""Enlaza cada SPK_xx diarizado con su nombre público y escribe speakers_<episodio>.csv.

Sesión de control del Congreso (configs/congreso_202.json): cada bloque tiene dos personas con
nombre y tratamiento oficial (Diputada Dª / Diputado D., Ministra / Ministro) y sus
intervenciones descargadas por separado. El género NO se infiere de la voz: la voz solo decide
qué SPK_xx es qué persona (coseno de embeddings WeSpeaker entre el hablante diarizado y sus
intervenciones oficiales). El género es el tratamiento oficial.

Quien preside (el SPK sin pareja con más voz) se marca rol=moderador/a; su género sale de cómo
la llaman los demás («señora presidenta» / «señor presidente»). Lo que no se enlaza queda U.
Detalle en data/annotations/speakers_<episodio>.link.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from explicometro import io as aio
from explicometro.schemas import Word

MIN_COS = 0.45
SPLIT_COS = 0.55
MAX_S = 60.0


def _norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def clean_spans(segs, spk):
    others = [(s.start, s.end) for s in segs if s.speaker != spk]
    out, total = [], 0.0
    for s in sorted((s for s in segs if s.speaker == spk), key=lambda s: s.end - s.start, reverse=True):
        a, b = s.start, s.end
        for oa, ob in others:
            if oa < b and ob > a:
                if oa <= a:
                    a = max(a, ob)
                else:
                    b = min(b, oa)
        if b - a >= 1.5:
            out.append((a, b))
            total += b - a
            if total >= MAX_S:
                break
    return out


def embed(inf, wav, spans):
    import soundfile as sf
    from pyannote.core import Segment

    dur = sf.info(str(wav)).duration
    vecs, ws = [], []
    for a, b in spans:
        b = min(b, dur - 0.05)
        if b - a < 1.0:
            continue
        v = np.asarray(inf.crop(str(wav), Segment(a, b))).reshape(-1)
        if np.isfinite(v).all():
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
            ws.append(b - a)
    if not vecs:
        return None
    m = np.average(np.stack(vecs), axis=0, weights=ws)
    return m / (np.linalg.norm(m) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--manifest", type=Path, default=Path("configs/congreso_202.json"))
    ap.add_argument("--interim", type=Path, default=Path("data/interim"))
    ap.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    a = ap.parse_args()

    import soundfile as sf
    import torch
    from pyannote.audio import Inference, Model

    man = json.loads(a.manifest.read_text(encoding="utf-8"))
    block = next(b for b in man["blocks"] if b["episode"] == a.episode)
    wd = a.interim / a.episode
    wav = wd / "audio.wav"
    segs = aio.read_rttm(wd / "diarization.rttm")
    words = aio.read_models(wd / "transcript.json", Word)

    from explicometro.diarize import allow_pyannote_checkpoints

    allow_pyannote_checkpoints()
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inf = Inference(model, window="whole")
    if torch.cuda.is_available():
        inf.to(torch.device("cuda"))

    people = {}
    for iv in block["interventions"]:
        p = people.setdefault(iv["name"], {"gender": iv["gender"], "spans": []})
        f = Path("data/raw/congreso_202_iv") / f"{iv['id']}.wav"
        dur = sf.info(str(f)).duration
        if dur > 4:
            p["spans"].append((f, 1.5, min(dur - 1.5, 45.0)))
    ref = {}
    for name, p in people.items():
        vs = [embed(inf, f, [(s, e)]) for f, s, e in p["spans"]]
        vs = [v for v in vs if v is not None]
        m = np.mean(vs, axis=0)
        ref[name] = m / (np.linalg.norm(m) + 1e-9)

    speech = defaultdict(float)
    for s in segs:
        speech[s.speaker] += s.end - s.start
    spk_emb = {spk: embed(inf, wav, clean_spans(segs, spk)) for spk in speech}
    cos = {
        spk: ({n: float(v @ r) for n, r in ref.items()} if v is not None else {})
        for spk, v in spk_emb.items()
    }

    assign = {}
    for name in ref:
        cands = sorted(
            ((c[name], spk) for spk, c in cos.items() if name in c and spk not in assign), reverse=True
        )
        if cands and cands[0][0] >= MIN_COS:
            assign[cands[0][1]] = name
    for spk, c in cos.items():
        if spk not in assign and c:
            name, val = max(c.items(), key=lambda kv: kv[1])
            if val >= SPLIT_COS:
                assign[spk] = name

    text = _norm(" ".join(w.text for w in sorted(words, key=lambda w: w.start)))
    n_fem = len(re.findall(r"senora presidenta", text))
    n_mas = len(re.findall(r"senor presidente", text))
    chair_gender = "M" if n_fem > n_mas else "H" if n_mas > n_fem else "U"
    rest = sorted((spk for spk in speech if spk not in assign), key=lambda s: -speech[s])
    chair = rest[0] if rest and speech[rest[0]] >= 3.0 else None

    rows = []
    for spk in sorted(speech):
        if spk in assign:
            rows.append(
                (
                    spk,
                    assign[spk],
                    people[assign[spk]]["gender"],
                    "invitada" if people[assign[spk]]["gender"] == "M" else "invitado",
                    "tratamiento oficial del Congreso; voz enlazada a sus intervenciones",
                )
            )
        elif spk == chair:
            rows.append(
                (
                    spk,
                    "Presidencia de la Cámara",
                    chair_gender,
                    "moderadora" if chair_gender == "M" else "moderador",
                    f"tratamiento en la sesión: senora presidenta x{n_fem}, senor presidente x{n_mas}",
                )
            )
        else:
            rows.append((spk, "", "U", "", "sin enlazar"))
    a.annotations.mkdir(parents=True, exist_ok=True)
    out = a.annotations / f"speakers_{a.episode}.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spk_id", "nombre_publico", "genero_publico", "rol", "fuente"])
        w.writerows(rows)
    (a.annotations / f"speakers_{a.episode}.link.json").write_text(
        json.dumps(
            {
                "speech_s": {k: round(v, 1) for k, v in speech.items()},
                "cos": cos,
                "assign": assign,
                "chair": chair,
                "chair_address_counts": {"senora_presidenta": n_fem, "senor_presidente": n_mas},
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(a.episode, rows)


if __name__ == "__main__":
    main()
