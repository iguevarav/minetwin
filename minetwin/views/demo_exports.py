"""Downloads for the fair demo, using the exact payload displayed in its iframe."""

import plotly.graph_objects as go
import streamlit as st

from minetwin.explanations import describe
from minetwin.presentation_ui import render_plot, render_table


def render_demo_exports(payload: dict) -> None:
    with st.expander("Descargas y explicación de las figuras y tablas de la demo"):
        st.caption(
            f"Fuente: {payload['fuente']} · modo {payload['modo']}. "
            "Las descargas usan los mismos valores de la demostración."
        )
        choices = [f"Camión {truck['id']}" for truck in payload["camiones"]] + [
            "Toda la flota",
            "Federación",
        ]
        selected = st.selectbox(
            "Contenido para descargar", choices, key="demo_export_section"
        )
        index = choices.index(selected)
        if index < len(payload["camiones"]):
            _truck(payload, payload["camiones"][index])
        elif selected == "Toda la flota":
            _fleet(payload)
        else:
            _federation(payload)


def _notes(payload: dict, kind: str, title: str, context: str = ""):
    return describe(
        kind,
        title,
        source=f"{payload['fuente']} · modo {payload['modo']} · {context}",
    )


def _table(payload, rows, kind, title, key, context=""):
    render_table(
        rows,
        explanation=_notes(payload, kind, title, context),
        key=f"demo_{key}",
        hide_index=True,
        width="stretch",
    )


def _plot(payload, figure, kind, title, key, context=""):
    figure.update_layout(template="plotly_white", height=360)
    render_plot(
        figure,
        explanation=_notes(payload, kind, title, context),
        key=f"demo_{key}",
    )


def _truck(payload: dict, truck: dict) -> None:
    identity = str(truck["id"])
    context = f"Camión {identity} · nodo {truck['nodo']} · fold {truck['grupo']}"
    readings = truck["lecturas"]
    sensors = [sensor for sensor in payload["sensores"] if sensor["clave"] != "perfil"]
    rows = [
        {
            "Readout": row["n"],
            "Paso temporal": row["t"],
            **{sensor["codigo"]: row["valores"][sensor["clave"]] for sensor in sensors},
            "Variables ausentes": row["sin_dato"],
            **{
                f"Perfil rango {i + 1}": row["perfil"][i] if row["perfil"] else None
                for i in range(6)
            },
        }
        for row in readings
    ]
    _table(
        payload,
        rows,
        "telemetry",
        f"Lecturas del camión {identity}",
        f"readings_{identity}",
        context,
    )
    for sensor in sensors:
        figure = go.Figure(
            go.Scatter(
                x=[row["t"] for row in readings],
                y=[row["valores"][sensor["clave"]] for row in readings],
                mode="lines+markers",
                name=sensor["codigo"],
                connectgaps=False,
            )
        )
        figure.update_layout(
            xaxis_title="Paso temporal",
            yaxis_title=f"Valor original {sensor['codigo']}",
        )
        start, end = truck["ventana"]["inicio"], truck["ventana"]["fin"]
        figure.add_vrect(
            x0=readings[start]["t"],
            x1=readings[end]["t"],
            fillcolor="#d9971a",
            opacity=0.12,
            line_width=0,
        )
        _plot(
            payload,
            figure,
            "telemetry",
            f"Sensor {sensor['codigo']} · camión {identity}",
            f"sensor_{identity}_{sensor['clave']}",
            context + " · banda sombreada: ventana causal",
        )
    figure = go.Figure()
    for i in range(6):
        figure.add_scatter(
            x=[row["t"] for row in readings],
            y=[row["perfil"][i] if row["perfil"] else None for row in readings],
            name=f"Rango {i + 1}",
            mode="lines+markers",
            connectgaps=False,
        )
    figure.update_layout(
        xaxis_title="Paso temporal",
        yaxis_title="Participación del incremento",
        yaxis_tickformat=".0%",
    )
    _plot(
        payload,
        figure,
        "profile",
        f"Perfil 397 · camión {identity}",
        f"profile_{identity}",
        context,
    )
    rows = [
        {
            "Sensor": sensor["codigo"],
            **{
                stat["nombre"]: truck["resumen"][sensor["clave"]][stat["clave"]]
                for stat in payload["estadisticas"]
            },
        }
        for sensor in sensors
    ]
    _table(
        payload,
        rows,
        "window",
        f"Ventana causal · camión {identity}",
        f"window_{identity}",
        context,
    )
    layers = payload["modelo"]["capas"]
    architecture = go.Figure(
        go.Bar(x=[f"Capa {i + 1}" for i in range(len(layers))], y=layers)
    )
    architecture.update_layout(
        xaxis_title="Entrada, capas ocultas y salida", yaxis_title="Número de unidades"
    )
    _plot(
        payload,
        architecture,
        "architecture",
        "Dimensiones del MLP demostrativo",
        f"architecture_{identity}",
        context,
    )
    rows = [
        {"Clase": i, "Significado": meaning, "Probabilidad": truck["probabilidades"][i]}
        for i, meaning in enumerate(payload["clases"])
    ]
    _table(
        payload,
        rows,
        "probabilities",
        f"Probabilidades · camión {identity}",
        f"probabilities_{identity}",
        context,
    )
    figure = go.Figure(
        go.Bar(x=[f"Clase {i}" for i in range(5)], y=truck["probabilidades"])
    )
    figure.update_layout(
        xaxis_title="Clase",
        yaxis_title="Probabilidad calibrada",
        yaxis_tickformat=".0%",
    )
    _plot(
        payload,
        figure,
        "probabilities",
        f"Probabilidades calibradas · camión {identity}",
        f"probability_plot_{identity}",
        context,
    )
    queue = [
        {"Bloque de cola": i + 1, "Probabilidad media clase 4": value}
        for i, value in enumerate(truck["fila"])
    ]
    _table(
        payload,
        queue,
        "queue",
        f"Distribución de la cola · camión {identity}",
        f"queue_{identity}",
        context
        + " · 100 bloques de probabilidades promediadas; no son vehículos individuales",
    )
    figure = go.Figure(
        go.Scatter(
            x=list(range(1, len(queue) + 1)),
            y=truck["fila"],
            mode="lines",
            name="Media del bloque",
        )
    )
    figure.add_hline(
        y=truck["umbral"], line_dash="dash", annotation_text="Umbral de revisión"
    )
    figure.add_hline(
        y=truck["probabilidades"][4],
        line_color="#d9971a",
        annotation_text=f"Camión {identity}",
    )
    figure.update_layout(
        xaxis_title="Bloque de cola (100 grupos)",
        yaxis_title="Probabilidad calibrada de clase 4",
        yaxis_tickformat=".0%",
    )
    _plot(
        payload,
        figure,
        "queue",
        f"Posición en la cola · camión {identity}",
        f"queue_plot_{identity}",
        context
        + " · curva agrupada en 100 bloques; líneas: umbral y probabilidad del camión",
    )
    rows = [
        {
            "Vehículo": identity,
            "Puesto": truck["puesto"],
            "Total del fold": truck["total_grupo"],
            "Capacidad": payload["capacidad"],
            "Probabilidad clase 4": truck["probabilidades"][4],
            "Umbral": truck["umbral"],
            "Prioritario": truck["prioritario"],
        }
    ]
    _table(
        payload,
        rows,
        "queue",
        f"Decisión de prioridad · camión {identity}",
        f"decision_{identity}",
        context,
    )


def _fleet(payload: dict) -> None:
    rows = payload["flota"]["capacidades"]
    context = f"Promedio de {payload['grupos']} folds fuera de fold en train"
    _table(
        payload, rows, "capacity", "Capacidad de revisión de la flota", "fleet", context
    )
    figure = go.Figure()
    for field, label in (("encontrados", "Priorización"), ("azar", "Al azar")):
        figure.add_scatter(
            x=[row["revisiones"] for row in rows],
            y=[row[field] for row in rows],
            name=label,
            mode="lines+markers",
        )
    figure.update_layout(
        xaxis_title="Capacidad de revisión", yaxis_title="Casos urgentes encontrados"
    )
    _plot(
        payload,
        figure,
        "capacity",
        "Detección frente a revisión al azar",
        "fleet_plot",
        context,
    )


def _federation(payload: dict) -> None:
    federation = payload.get("federacion")
    if not federation:
        st.info("El payload mostrado no contiene resultados de federación.")
        return
    rows = [{"Concepto": key, "Valor": value} for key, value in federation.items()]
    _table(
        payload, rows, "federation", "Transferencia federada de la demo", "federation"
    )
    figure = go.Figure(
        go.Bar(
            x=["Estados publicados", "Centralización de variables"],
            y=[federation["mb_compartidos"], federation["mb_centralizado"]],
        )
    )
    figure.update_layout(
        xaxis_title="Modalidad de transferencia", yaxis_title="MB decimales"
    )
    _plot(
        payload,
        figure,
        "federation",
        "Volumen compartido en la federación",
        "federation_plot",
    )
