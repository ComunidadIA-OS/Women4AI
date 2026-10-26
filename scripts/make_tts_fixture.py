"""Genera una conversación sintética de dos voces con la verdad conocida.

Salida en ``--out`` (por defecto tests/fixtures/tts_dialogue/):
- ``dialogue.wav``: 16 kHz mono PCM16.
- ``truth.rttm``: intervalos exactos de habla por hablante (SPK_00 = A, SPK_01 = B).
- ``truth.json``: lista de intervenciones ``{speaker, start, end, text}``.

El guion incluye turnos normales, una interrupción competitiva (B entra 1,2 s antes de que
A termine), un asentimiento corto solapado y silencios largos.

Motores TTS:
- ``styletts2`` (recomendado, en la máquina GPU): BSC-LT/styletts2-spanish-multispeaker vía
  un comando externo, p. ej.::

      --tts styletts2 --tts-cmd "python infer.py --text {text} --speaker {voice} --out {out}" \\
      --voices A=<id_voz_masculina>,B=<id_voz_femenina>

  (el comando depende de cómo tengáis instalado StyleTTS2; ``{text}``, ``{voice}`` y
  ``{out}`` se sustituyen, ``{out}`` debe acabar siendo un WAV).
- ``flite`` (cualquier máquina con ffmpeg compilado con libflite): voces ``kal`` y ``slt``.
  Pronuncia con fonética inglesa: vale para VAD, diarización y alineación, no para medir WER.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SR = 16000

# (hablante, texto, hueco respecto al FINAL de la intervención anterior; negativo = solape)
SCRIPT: list[tuple[str, str, float]] = [
    ("A", "Buenas tardes y bienvenidos a la tertulia de hoy sobre inteligencia artificial.", 0.6),
    ("B", "Gracias por invitarme. Creo que el tema de los datos abiertos es fundamental.", 0.8),
    ("A", "Estoy de acuerdo, aunque hay que pensar también en la privacidad de las personas.", 0.7),
    ("B", "Sí, claro.", -1.6),  # asentimiento corto en solape
    ("A", "Por eso los modelos deberían entrenarse con corpus públicos y bien documentados.", 0.4),
    ("B", "Perdona, pero eso ya lo explicamos nosotras en el informe del año pasado.", -1.2),  # interrupción
    ("A", "Bueno, en cualquier caso lo importante es medir quién habla y cuánto.", 2.5),
    ("B", "Exacto, y por eso proponemos métricas abiertas y reproducibles.", 1.0),
]

FLITE_VOICES = {"A": "kal", "B": "slt"}


@dataclass(frozen=True)
class Utterance:
    speaker: str  # SPK_00 / SPK_01
    start: float
    end: float
    text: str


def _read_wav(path: Path) -> np.ndarray:
    """Lee cualquier WAV vía ffmpeg y lo devuelve a 16 kHz mono float32."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "x.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
             "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(tmp)],
            check=True,
        )  # fmt: skip
        with wave.open(str(tmp), "rb") as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def trim(x: np.ndarray, rel_db: float = -40.0, frame: int = 160) -> np.ndarray:
    """Recorta silencio inicial y final (energía por tramos de 10 ms relativa al máximo)."""
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    idx = np.flatnonzero(db > rel_db)
    return x[idx[0] * frame : (idx[-1] + 1) * frame] if len(idx) else x


def synth_flite(text: str, voice: str, out: Path) -> None:
    safe = text.replace("'", "").replace(":", " ").replace("\\", " ")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"flite=text='{safe}':voice={voice}", "-ar", str(SR), "-ac", "1", str(out)],
        check=True,
        capture_output=True,  # flite escribe avisos de pronunciación por stderr
    )  # fmt: skip


def synth_command(text: str, voice: str, out: Path, template: str) -> None:
    cmd = template.format(text=shlex.quote(text), voice=shlex.quote(voice), out=shlex.quote(str(out)))
    subprocess.run(cmd, shell=True, check=True)


def build(out_dir: Path, tts: str = "flite", tts_cmd: str | None = None, voices: dict[str, str] | None = None,
          lead_s: float = 1.0, tail_s: float = 1.5) -> list[Utterance]:  # fmt: skip
    """Sintetiza el guion, lo mezcla con solapes y escribe wav + verdad. Devuelve la verdad."""
    out_dir.mkdir(parents=True, exist_ok=True)
    voices = voices or (FLITE_VOICES if tts == "flite" else {})
    spk_id = {"A": "SPK_00", "B": "SPK_01"}
    clips: list[tuple[str, str, np.ndarray]] = []
    with tempfile.TemporaryDirectory() as d:
        for i, (spk, text, _gap) in enumerate(SCRIPT):
            raw = Path(d) / f"u{i}.wav"
            if tts == "flite":
                synth_flite(text, voices[spk], raw)
            elif tts == "styletts2":
                if not tts_cmd or spk not in voices:
                    raise SystemExit("styletts2 necesita --tts-cmd y --voices A=..,B=..")
                synth_command(text, voices[spk], raw, tts_cmd)
            else:
                raise SystemExit(f"TTS desconocido: {tts}")
            clips.append((spk, text, trim(_read_wav(raw))))

    truth: list[Utterance] = []
    t = lead_s
    for (spk, text, clip), (_s, _t, gap) in zip(clips, SCRIPT, strict=True):
        start = t + gap if truth else lead_s
        start = max(0.0, start)
        end = start + len(clip) / SR
        truth.append(Utterance(spk_id[spk], round(start, 3), round(end, 3), text))
        # el siguiente hueco se mide desde el final de la intervención principal en curso
        t = max(t, end) if gap < 0 and len(clip) / SR < 1.5 else end

    total = int((max(u.end for u in truth) + tail_s) * SR)
    mix = np.zeros(total, dtype=np.float32)
    for u, (_spk, _text, clip) in zip(truth, clips, strict=True):
        i0 = round(u.start * SR)
        mix[i0 : i0 + len(clip)] += clip / (np.abs(clip).max() + 1e-9) * 0.5
    mix /= max(1.0, float(np.abs(mix).max()) / 0.95)
    pcm = (mix * 32767).astype("<i2")
    with wave.open(str(out_dir / "dialogue.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    rttm = [
        f"SPEAKER dialogue 1 {u.start:.3f} {u.end - u.start:.3f} <NA> <NA> {u.speaker} <NA> <NA>"
        for u in truth
    ]
    (out_dir / "truth.rttm").write_text("\n".join(rttm) + "\n", encoding="utf-8")
    meta = {"tts": tts, "voices": voices, "utterances": [asdict(u) for u in truth]}
    (out_dir / "truth.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return truth


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("tests/fixtures/tts_dialogue"))
    p.add_argument("--tts", choices=["flite", "styletts2"], default="flite")
    p.add_argument("--tts-cmd", help="Plantilla de comando para styletts2 ({text} {voice} {out})")
    p.add_argument("--voices", help="A=voz,B=voz")
    a = p.parse_args()
    voices = dict(kv.split("=", 1) for kv in a.voices.split(",")) if a.voices else None
    truth = build(a.out, a.tts, a.tts_cmd, voices)
    print(f"{a.out}/dialogue.wav: {len(truth)} intervenciones, {truth[-1].end:.1f} s")


if __name__ == "__main__":
    main()
