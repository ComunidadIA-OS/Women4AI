"""Género de la Presidencia a nivel de sesión (Congreso, sesión 202).

En cada bloque, quien preside se llama «señora presidenta» o «señor presidente» solo a veces.
Se toman como referencia los bloques donde el tratamiento es claro (excluyendo «presidente del
Gobierno») y, en los demás, se asigna ese tratamiento solo si la voz de quien preside es la
misma (coseno >= 0,6 de embeddings WeSpeaker). Si no, queda U. Reescribe speakers_<ep>.csv.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from explicometro import io as aio
from explicometro.schemas import Word
from link_speakers import clean_spans, embed

SAME_VOICE = 0.6


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def main():
    import torch
    from pyannote.audio import Inference, Model

    from explicometro.diarize import allow_pyannote_checkpoints

    allow_pyannote_checkpoints()
    inf = Inference(Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM"), window="whole")
    if torch.cuda.is_available():
        inf.to(torch.device("cuda"))
    man = json.loads(Path("configs/congreso_202.json").read_text(encoding="utf-8"))
    info = {}
    for b in man["blocks"]:
        ep = b["episode"]
        csvp = Path(f"data/annotations/speakers_{ep}.csv")
        if not csvp.exists():
            continue
        rows = list(csv.DictReader(csvp.open(encoding="utf-8")))
        chair = next((r["spk_id"] for r in rows if r["rol"].startswith("moderador")), None)
        if not chair:
            continue
        wd = Path("data/interim") / ep
        segs = aio.read_rttm(wd / "diarization.rttm")
        text = norm(
            " ".join(
                w.text for w in sorted(aio.read_models(wd / "transcript.json", Word), key=lambda w: w.start)
            )
        )
        f = len(re.findall(r"senora presidenta", text))
        m = len(re.findall(r"senor presidente(?! del gobierno)", text))
        own = "M" if f > m else "H" if m > f else None
        info[ep] = {
            "rows": rows,
            "chair": chair,
            "f": f,
            "m": m,
            "own": own,
            "emb": embed(inf, wd / "audio.wav", clean_spans(segs, chair)),
        }
    refs = [(ep, d) for ep, d in info.items() if d["own"] and d["emb"] is not None]
    out = {}
    for ep, d in info.items():
        g, src = (
            d["own"],
            f"tratamiento en el bloque: senora presidenta x{d['f']}, senor presidente x{d['m']}",
        )
        if g is None and d["emb"] is not None and refs:
            sims = sorted(((float(d["emb"] @ r["emb"]), rep, r["own"]) for rep, r in refs), reverse=True)
            # voto de las referencias con la misma voz
            same = [(s, rep, rg) for s, rep, rg in sims if s >= SAME_VOICE]
            gs = {rg for _, _, rg in same}
            if len(gs) == 1:
                g = gs.pop()
                src = (
                    f"misma voz (coseno {same[0][0]:.2f}) que la Presidencia de {same[0][1]}, "
                    "donde consta el tratamiento"
                )
        g = g or "U"
        for r in d["rows"]:
            if r["spk_id"] == d["chair"]:
                r["genero_publico"], r["fuente"] = g, src
                r["rol"] = "moderadora" if g == "M" else "moderador"
        p = Path(f"data/annotations/speakers_{ep}.csv")
        with p.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["spk_id", "nombre_publico", "genero_publico", "rol", "fuente"])
            w.writeheader()
            w.writerows(d["rows"])
        out[ep] = {"chair": d["chair"], "genero": g, "fuente": src}
    Path("data/annotations/presidencia_202.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
