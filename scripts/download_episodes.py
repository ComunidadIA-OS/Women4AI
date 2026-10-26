"""Descarga episodios listados en un manifiesto YAML y verifica sus hashes.

El audio NUNCA se versiona: va a ``data/raw/`` (ignorado por git). Lo que sí se publica
es el manifiesto y ``configs/episodes.sha256`` para que cualquiera reproduzca el corpus.

Manifiesto (ver configs/episodes.example.yaml)::

    episodes:
      - id: programa_2026-10-01
        url: https://...
        downloader: http        # http | yt-dlp
        sha256: null            # se rellena tras la primera descarga verificada
        source: "RTVE Play"
        license: "uso para investigación"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

from explicometro.ingest import sha256_file


def download(url: str, dst: Path, downloader: str) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if downloader == "http":
        with urllib.request.urlopen(url) as r, dst.open("wb") as f:
            shutil.copyfileobj(r, f)
        return dst
    if downloader == "yt-dlp":
        if shutil.which("yt-dlp") is None:
            raise RuntimeError("yt-dlp no está instalado")
        out = dst.with_suffix(".%(ext)s")
        subprocess.run(["yt-dlp", "-x", "--audio-format", "wav", "-o", str(out), url], check=True)
        return dst.with_suffix(".wav")
    raise ValueError(f"downloader desconocido: {downloader}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("configs/episodes.yaml"))
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--sums", type=Path, default=Path("configs/episodes.sha256"))
    p.add_argument("--only", nargs="*", help="IDs a descargar")
    a = p.parse_args()

    if not a.manifest.exists():
        print(f"No existe {a.manifest}. Copia configs/episodes.example.yaml.", file=sys.stderr)
        return 1
    episodes = (yaml.safe_load(a.manifest.read_text(encoding="utf-8")) or {}).get("episodes", [])
    sums: dict[str, str] = {}
    failed = 0
    for ep in episodes:
        if a.only and ep["id"] not in a.only:
            continue
        downloader = ep.get("downloader", "http")
        suffix = ".wav" if downloader == "yt-dlp" else (Path(ep["url"]).suffix or ".bin")
        dst = a.out / f"{ep['id']}{suffix}"
        if not dst.exists():
            print(f"↓ {ep['id']}")
            dst = download(ep["url"], dst, downloader)
        digest = sha256_file(dst)
        expected = ep.get("sha256")
        if expected and expected != digest:
            print(f"✗ {ep['id']}: sha256 no coincide ({digest} != {expected})", file=sys.stderr)
            failed += 1
            continue
        sums[dst.name] = digest
        print(f"✓ {ep['id']} {digest}")
    if sums:
        lines = [f"{h}  {name}" for name, h in sorted(sums.items())]
        a.sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
