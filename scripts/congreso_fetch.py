"""Descarga los vídeos oficiales de la sesión de control (Congreso, sesión 202, 30/09/2026).

Lee configs/congreso_202.json, descarga cada mp4 de static.congreso.es (enlace de descarga del
archivo audiovisual), extrae audio 16 kHz mono a data/raw/<episodio>.wav (bloques = preguntas)
y data/raw/congreso_202_iv/<id>.wav (intervenciones, solo para enlazar voz y nombre público) y
escribe configs/episodes.yaml con el hash SHA-256 de cada vídeo. El audio no se versiona.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dst: Path) -> Path:
    if not dst.exists():
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def to_wav(src: Path, dst: Path) -> None:
    if not dst.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(dst),
            ],
            check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("configs/congreso_202.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/raw/_mp4"))
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text(encoding="utf-8"))
    a.cache.mkdir(parents=True, exist_ok=True)
    ivdir = Path("data/raw/congreso_202_iv")
    ivdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for b in man["blocks"]:
        jobs.append((b["block_id"], Path("data/raw") / f"{b['episode']}.wav"))
        for iv in b["interventions"]:
            jobs.append((iv["id"], ivdir / f"{iv['id']}.wav"))

    def work(job):
        i, wav = job
        mp4 = fetch(man["url_template"].format(id=i), a.cache / f"{i}.mp4")
        to_wav(mp4, wav)
        return i, sha256(mp4)

    with ThreadPoolExecutor(8) as ex:
        hashes = dict(ex.map(work, jobs))
    eps = []
    for b in man["blocks"]:
        eps.append(
            {
                "id": b["episode"],
                "url": man["url_template"].format(id=b["block_id"]),
                "downloader": "http",
                "sha256": hashes[b["block_id"]],
                "source": "Congreso de los Diputados (archivo audiovisual, versión original)",
                "license": (
                    "Reutilización permitida citando la fuente y sin alterar el contenido "
                    "(aviso legal de congreso.es)"
                ),
                "channel": "Congreso",
                "program": "Pleno - sesión de control",
                "date": "2026-09-30",
                "year": 2026,
                "title": b["title"],
            }
        )
    Path("configs/episodes.yaml").write_text(
        yaml.safe_dump({"episodes": eps}, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"ok: {len(eps)} episodios, {len(jobs)} ficheros")


if __name__ == "__main__":
    main()
