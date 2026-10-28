"""Muestra qué dice cada hablante (SPK_xx) para rellenar ``speakers_<episodio>.csv``.

Lee ``data/interim/<episodio>/transcript.json`` (tras ``--until align``) y, por hablante,
imprime su tiempo de habla y las primeras frases con su minuto, para escucharlas en el audio.
Si no existe el CSV, escribe una plantilla con género ``U`` para editar.

    python scripts/speaker_samples.py --episode ep01
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from explicometro.config import load_config


def utterances(words: list[dict], max_gap_s: float = 1.0) -> dict[str, list[tuple[float, str]]]:
    """Agrupa palabras consecutivas del mismo hablante en frases (inicio, texto)."""
    out: dict[str, list[tuple[float, str]]] = defaultdict(list)
    cur_spk, cur_start, cur_end, buf = None, 0.0, 0.0, []
    for w in sorted(words, key=lambda w: w["start"]):
        spk = w.get("speaker")
        if spk != cur_spk or w["start"] - cur_end > max_gap_s:
            if cur_spk and buf:
                out[cur_spk].append((cur_start, " ".join(buf)))
            cur_spk, cur_start, buf = spk, w["start"], []
        buf.append(w["text"].strip())
        cur_end = w["end"]
    if cur_spk and buf:
        out[cur_spk].append((cur_start, " ".join(buf)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("-n", type=int, default=3, help="Frases por hablante")
    a = ap.parse_args()

    cfg = load_config(a.config)
    words = json.loads((cfg.paths.interim / a.episode / "transcript.json").read_text(encoding="utf-8"))
    durations: dict[str, float] = defaultdict(float)
    for w in words:
        if w.get("speaker"):
            durations[w["speaker"]] += w["end"] - w["start"]
    utts = utterances(words)
    for spk in sorted(utts):
        print(f"\n{spk}  ({durations[spk] / 60:.1f} min de palabras, {len(utts[spk])} intervenciones)")
        long_ones = [u for u in utts[spk] if len(u[1].split()) >= 6] or utts[spk]
        for start, text in long_ones[: a.n]:
            print(f"  [{int(start // 60):02d}:{int(start % 60):02d}] {text[:160]}")

    csv_path = Path(cfg.paths.annotations) / f"speakers_{a.episode}.csv"
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = "".join(f"{spk},,U,\n" for spk in sorted(utts))
        csv_path.write_text("spk_id,nombre_publico,genero_publico,rol\n" + rows, encoding="utf-8")
        print(f"\nPlantilla escrita en {csv_path}: cambia U por H o M (solo si es público).")
    else:
        print(f"\n{csv_path} ya existe; no se toca.")


if __name__ == "__main__":
    main()
