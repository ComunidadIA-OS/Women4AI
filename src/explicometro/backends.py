"""Registro mínimo de backends por etapa."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., object])


class BackendError(RuntimeError):
    pass


def resolve(backends: Mapping[str, F], name: str, stage: str) -> F:
    try:
        return backends[name]
    except KeyError:
        raise BackendError(
            f"Backend '{name}' no existe para la etapa '{stage}'. Disponibles: {sorted(backends)}"
        ) from None


def not_implemented(stage: str, backend: str, hint: str) -> BackendError:
    """Error uniforme para backends reales aún no implementados."""
    return BackendError(
        f"[{stage}] backend '{backend}' pendiente de implementar. {hint} "
        "Usa --dry-run (stubs) mientras tanto."
    )
