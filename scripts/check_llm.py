"""Comprueba el endpoint LLM (OVHcloud AI Endpoints o vLLM) antes de lanzar el pipeline.

1. Lista los modelos que ofrece el endpoint (``/models``).
2. Juzga un par sintético con el prompt real del proyecto y valida el JSON.

Lee ``LLM_BASE_URL``, ``LLM_API_KEY`` y ``LLM_MODEL`` de ``.env``. Nunca imprime la clave.

    python scripts/check_llm.py              # lista modelos y prueba LLM_MODEL
    python scripts/check_llm.py --list       # solo lista modelos
"""

from __future__ import annotations

import argparse
import json
import sys

from explicometro.config import env, load_config
from explicometro.llm import LlmParams, judge, llm_settings, load_prompt, make_client, render_prompt

A = "Creo que la subida del salario mínimo ha tenido un efecto positivo en el empleo juvenil."
B = (
    "Bueno, a ver, déjame que te explique cómo funciona esto realmente, porque no es así. "
    "El salario mínimo es un precio y cuando subes un precio baja la demanda, eso lo sabe "
    "cualquiera que haya estudiado un poco de economía básica."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--list", action="store_true", help="Solo lista los modelos")
    a = ap.parse_args()

    cfg = load_config(a.config)
    p = cfg.stages.llm.parsed(LlmParams)
    settings = llm_settings(p)
    print(f"Endpoint: {settings['base_url']}")
    print(f"Modelo:   {settings['model']}")
    print(f"Clave:    {'definida' if env('LLM_API_KEY') else 'NO definida (LLM_API_KEY en .env)'}")
    client = make_client(settings, p)

    try:
        ids = sorted(m.id for m in client.models.list())
        print(f"\nModelos disponibles ({len(ids)}):")
        for i in ids:
            print(f"  - {i}{'   <== LLM_MODEL' if i == settings['model'] else ''}")
        if settings["model"] not in ids:
            print(f"\n⚠️  '{settings['model']}' no está en la lista: cambia LLM_MODEL en .env.")
    except Exception as e:  # algunos servidores no exponen /models
        print(f"\n(no se pudo listar /models: {e})")
    if a.list:
        return 0

    prompt = render_prompt(load_prompt(p.prompt), A, B)
    rec = judge(client, settings["model"], prompt, B, p, json_mode=p.json_mode)
    print(f"\nPrueba de juicio: ok={rec['ok']} intentos={rec['attempts']} latencia={rec['latency_s']:.2f} s")
    print(f"tokens: entrada={rec['prompt_tokens']} salida={rec['completion_tokens']}")
    print(json.dumps(rec["output"] if rec["ok"] else {"error": rec["error"], "raw": rec.get("raw")},
                     ensure_ascii=False, indent=2))  # fmt: skip
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
