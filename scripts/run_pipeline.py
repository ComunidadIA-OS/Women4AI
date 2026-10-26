"""Ejecuta el pipeline. Equivale a ``python -m explicometro run ...``."""

import sys

from explicometro.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
