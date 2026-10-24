"""Display an explanation and a download for every application table/figure."""

from dataclasses import replace
from functools import partial
from hashlib import sha256

import numpy as np
import streamlit as st

from minetwin.exports import (
    XLSX_MIME,
    Explanation,
    explained_rows,
    filename,
    plotly_jpg,
    table_xlsx,
)


def _context(explanation: Explanation) -> Explanation:
    keys = ("scania_navigation", "scania_split", "scania_node", "scania_vehicle")
    context = " · ".join(
        str(st.session_state[key]) for key in keys if key in st.session_state
    )
    return replace(
        explanation,
        source=explanation.source
        + (f" · Selección de interfaz: {context}" if context else ""),
    )


def render_table(rows: list[dict], *, explanation: Explanation, key: str, **kwargs):
    explanation = explained_rows(_context(explanation), rows)
    st.dataframe(rows, **kwargs)
    st.caption(explanation.markdown())
    st.download_button(
        "Descargar tabla en Excel",
        data=partial(table_xlsx, rows, explanation),
        file_name=f"{filename(explanation.title)}.xlsx",
        mime=XLSX_MIME,
        key=f"excel_{key}",
        on_click="ignore",
    )


def _figure_result(figure) -> str:
    observations = []
    for trace in figure.data:
        if trace.type == "heatmap":
            values = np.asarray(trace.z, dtype=float)
            observations.append(
                f"{values.shape[0]} filas × {values.shape[1]} columnas; "
                f"{np.nansum(values):.6g} observaciones en total"
            )
        else:
            horizontal = getattr(trace, "orientation", None) == "h"
            numbers = trace.x if horizontal else trace.y
            labels = trace.y if horizontal else trace.x
            valid = []
            for index, value in enumerate(numbers if numbers is not None else []):
                if value is not None and np.isfinite(float(value)):
                    label = labels[index] if labels is not None else index + 1
                    valid.append((label, float(value)))
            if valid:
                peak, maximum = max(valid, key=lambda point: point[1])
                observations.append(
                    f"{trace.name or 'Serie'}: {len(valid)} puntos válidos; "
                    f"máximo {maximum:.6g} en {peak}; "
                    f"inicio {valid[0][1]:.6g}, final {valid[-1][1]:.6g}"
                )
    return "; ".join(observations) or "Sin puntos numéricos disponibles."


@st.cache_data(max_entries=24, show_spinner=False)
def _jpg(serialized: str, explanation: Explanation) -> bytes:
    import plotly.io as pio

    return plotly_jpg(pio.from_json(serialized), explanation)


def render_plot(figure, *, explanation: Explanation, key: str | None = None):
    explanation = replace(_context(explanation), result=_figure_result(figure))
    serialized = figure.to_json()
    identity = key or sha256((explanation.title + serialized).encode()).hexdigest()[:16]
    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
        key=f"plot_{identity}",
    )
    st.caption(explanation.markdown())
    # Streamlit calls the producer only when requested; browsing never rasterizes.
    st.download_button(
        "Descargar imagen JPG · 600 DPI",
        data=partial(_jpg, serialized, explanation),
        file_name=f"{filename(explanation.title)}.jpg",
        mime="image/jpeg",
        key=f"jpg_{identity}",
        on_click="ignore",
    )
