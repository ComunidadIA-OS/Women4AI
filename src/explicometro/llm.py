"""C3 (explicación no solicitada) y C4 (corrección directa) con un LLM abierto.

Cliente OpenAI-compatible: ``LLM_BASE_URL``, ``LLM_API_KEY`` y ``LLM_MODEL`` (.env) tienen
prioridad sobre ``stages.llm.params``. La salida del LLM se valida como JSON y cada
``Event.evidence`` debe ser una cita literal del turno etiquetado.
"""

from __future__ import annotations

from .backends import not_implemented, resolve
from .config import RuntimeCfg, StageCfg, env
from .schemas import Event, Speaker, Turn


def llm_settings(cfg: StageCfg) -> dict[str, str | None]:
    """Resuelve endpoint y modelo (.env > YAML). No devuelve la clave en logs."""
    return {
        "base_url": env("LLM_BASE_URL", cfg.params.get("base_url")),
        "model": env("LLM_MODEL", cfg.params.get("model")),
    }


def _stub(turns: list[Turn], speakers: list[Speaker], cfg: StageCfg, rt: RuntimeCfg) -> list[Event]:
    return []


def _openai_compatible(
    turns: list[Turn], speakers: list[Speaker], cfg: StageCfg, rt: RuntimeCfg
) -> list[Event]:
    raise not_implemented("llm", "openai_compatible", "Define LLM_BASE_URL/LLM_API_KEY/LLM_MODEL en .env.")


BACKENDS = {"stub": _stub, "openai_compatible": _openai_compatible}


def run(turns: list[Turn], speakers: list[Speaker], cfg: StageCfg, rt: RuntimeCfg) -> list[Event]:
    return resolve(BACKENDS, cfg.backend, "llm")(turns, speakers, cfg, rt)
