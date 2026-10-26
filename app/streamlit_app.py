"""Visor mínimo de resultados: ``make app``.

Lee solo artefactos del contrato (metrics.json, speakers.json, turns.parquet,
events.parquet) desde data/interim/<episodio>/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from explicometro import io as aio
from explicometro.config import load_config
from explicometro.report import fmt_ratio
from explicometro.schemas import EpisodeMetrics, Event, Speaker, Turn

CONFIG = Path("configs/default.yaml")

st.set_page_config(page_title="Explicómetro", layout="wide")
st.title("Explicómetro")

cfg = load_config(CONFIG if CONFIG.exists() else None)
episodes = sorted(p.parent.name for p in cfg.paths.interim.glob("*/metrics.json"))
if not episodes:
    st.info(f"No hay episodios procesados en {cfg.paths.interim}. Prueba `make dry-run`.")
    st.stop()

ep = st.sidebar.selectbox("Episodio", episodes)
d = cfg.paths.interim / ep
m = aio.read_model(d / "metrics.json", EpisodeMetrics)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Duración (s)", f"{m.duration_s:.0f}")
c2.metric("Habla (s)", f"{m.speech_s:.0f}")
c3.metric("Hablantes", m.n_speakers)
r = m.asymmetry_ratio
c4.metric("Ratio de asimetría", f"{r.value:.2f}" if r.value is not None else r.status, help=fmt_ratio(r))
if m.dry_run:
    st.warning("Resultado de dry-run (stubs): las métricas están vacías por diseño.")

st.subheader("A. Presencia por género")
st.dataframe(
    pd.DataFrame({g.value: s.model_dump() for g, s in m.presence.by_gender.items()}).T,
    width="stretch",
)

u = m.unknown
st.caption(
    f"Fuera de las métricas por género (U): {u.speakers} hablantes, {u.time_s:.0f} s"
    + ("" if u.time_share is None else f" ({u.time_share:.0%} del tiempo de habla)")
    + ". Ver docs/gender.md."
)

st.subheader("C. Conductas por díada")
rows = [
    {"tipo": t.value, "díada": dy.value, **s.model_dump()}
    for t, dyads in m.events.items()
    for dy, s in dyads.items()
]
st.dataframe(pd.DataFrame(rows), width="stretch")
st.caption(
    f"Exceso de interrupción: {fmt_ratio(m.interruption_excess)} · "
    f"asimetría: {fmt_ratio(m.asymmetry_ratio)}. Eventos del moderador aparte: "
    f"{sum(s.count for row in m.moderator_events.values() for s in row.values())}."
)

with st.expander("Hablantes, turnos y eventos"):
    if (d / "speakers.json").exists():
        st.dataframe(pd.DataFrame([s.model_dump() for s in aio.read_models(d / "speakers.json", Speaker)]))
    if (d / "turns.parquet").exists():
        st.dataframe(pd.DataFrame([t.model_dump() for t in aio.read_parquet(d / "turns.parquet", Turn)]))
    if (d / "events.parquet").exists():
        st.dataframe(pd.DataFrame([e.model_dump() for e in aio.read_parquet(d / "events.parquet", Event)]))
