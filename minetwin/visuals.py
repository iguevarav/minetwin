import plotly.graph_objects as go

from minetwin.domain import EngineStatus, OperatingState
from minetwin.services import MineTwin

AMBER = "#b87500"
MUTED = "#5f6b7a"
STATUS_COLORS = {
    EngineStatus.NORMAL: "#157347",
    EngineStatus.WATCH: "#946200",
    EngineStatus.ALERT: "#c73e3e",
    EngineStatus.UNKNOWN: MUTED,
}
ROUTE_STOPS = (
    (OperatingState.WAITING, 115, 245, "Espera"),
    (OperatingState.LOADING, 205, 100, "Carga"),
    (OperatingState.HAULING, 430, 85, "Traslado cargado"),
    (OperatingState.UNLOADING, 555, 235, "Descarga"),
    (OperatingState.RETURNING, 340, 315, "Retorno vacío"),
)


def operation_map(simulation: MineTwin) -> str:
    twin = simulation.twin
    state = twin.observation.operating_state if twin else None
    color = STATUS_COLORS[twin.diagnosis.status] if twin else MUTED
    markers = []
    for stop_state, x, y, label in ROUTE_STOPS:
        active = stop_state == state
        marker_color = color if active else "#5e6671"
        markers.append(
            f'<circle cx="{x}" cy="{y}" r="5" fill="{marker_color}"/>'
            f'<text x="{x}" y="{y + 30}" text-anchor="middle" '
            f'fill="{color if active else MUTED}" font-size="13">{label}</text>'
        )
        if active:
            markers.append(
                f'<g transform="translate({x}, {y})">'
                f'<circle r="23" fill="{color}" fill-opacity=".1" '
                f'stroke="{color}" stroke-opacity=".45"/>'
                f'<rect x="-13" y="-9" width="18" height="13" rx="2" fill="{color}"/>'
                f'<path d="M5 -5H10L14 1V5H5Z" fill="{color}"/>'
                '<circle cx="-8" cy="7" r="3.5" fill="#ffffff" stroke="currentColor"/>'
                '<circle cx="9" cy="7" r="3.5" fill="#ffffff" stroke="currentColor"/>'
                "</g>"
            )
    return (
        '<svg class="mt-map" viewBox="0 0 680 390" role="img" '
        'aria-labelledby="map-title map-description">'
        '<title id="map-title">Ciclo de acarreo</title>'
        '<desc id="map-description">Esquema referencial, sin coordenadas reales. '
        "Espera, carga, traslado cargado, descarga y retorno vacío.</desc>"
        '<defs><pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">'
        '<circle cx="1" cy="1" r=".7" fill="#dbe2ea"/></pattern></defs>'
        '<rect width="680" height="390" fill="url(#grid)"/>'
        '<g fill="none" stroke="#ccd4dd" stroke-width="1.2">'
        '<path d="M245 142 Q340 100 428 155T463 238Q369 292 264 252T245 142Z"/>'
        '<path d="M261 159Q343 120 412 166T443 226Q365 272 278 238T261 159Z"/>'
        '<path d="M280 174Q345 142 398 180T424 215Q361 249 295 226T280 174Z"/>'
        '<path d="M302 189Q348 163 384 192T399 212Q353 229 319 214T302 189Z"/>'
        "</g>"
        '<text x="346" y="209" text-anchor="middle" fill="#7a8490" '
        'font-size="10" letter-spacing="3">TAJO</text>'
        '<path d="M115 245L205 100L430 85L555 235L340 315Z" '
        'fill="none" stroke="#e4dccd" stroke-width="18" stroke-linejoin="round"/>'
        '<path d="M115 245L205 100L430 85L555 235L340 315Z" '
        'fill="none" stroke="#a36b15" stroke-width="2" stroke-dasharray="6 8"/>'
        + "".join(markers)
        + "</svg>"
    )


def signal_chart(simulation: MineTwin, signal: str, label: str, unit: str) -> go.Figure:
    history = simulation.history
    component = (
        "tire:" + signal.removeprefix("tire_pressure_")
        if signal.startswith("tire_pressure_")
        else (
            "brakes" if signal in ("brake_temperature", "slope_percent") else "engine"
        )
    )
    times = [packet.source_time for packet in history]
    figure = go.Figure(
        go.Scatter(
            x=times,
            y=[packet.reading(signal).value for packet in history],
            name=label,
            mode="lines" if len(history) > 1 else "markers",
            connectgaps=False,
            line={"color": AMBER, "width": 2},
            marker={"color": AMBER, "size": 7},
        )
    )
    if signal in ("engine_temperature", "engine_vibration"):
        baseline_index = 0 if signal == "engine_temperature" else 1
        offset = 8.0 if baseline_index == 0 else 0.8
        figure.add_trace(
            go.Scatter(
                x=times,
                y=[
                    simulation.profile.engine_baseline(packet.operating_state)[
                        baseline_index
                    ]
                    + offset
                    for packet in history
                ],
                name="Umbral",
                mode="lines",
                line={"color": MUTED, "width": 1, "dash": "dot"},
            )
        )
    for alert in simulation.alerts:
        if (
            alert.component == component
            and times
            and times[0] <= alert.opened_at <= times[-1]
        ):
            figure.add_vline(
                x=alert.opened_at,
                line_color=STATUS_COLORS[EngineStatus.ALERT],
                line_dash="dash",
                line_width=1,
            )
    for event in simulation.events:
        if (
            event.kind == "maintenance_completed"
            and event.component == component
            and times
            and times[0] <= event.timestamp <= times[-1]
        ):
            figure.add_vline(
                x=event.timestamp, line_color="#157347", line_dash="dot", line_width=1
            )
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Segoe UI, sans-serif", "color": MUTED, "size": 12},
        title={"text": label, "font": {"color": "#1b2430", "size": 15}},
        xaxis={
            "title": "Tiempo simulado · UTC",
            "gridcolor": "#e1e6ec",
            "zeroline": False,
        },
        yaxis={"title": unit, "gridcolor": "#e1e6ec", "zeroline": False},
        margin={"l": 20, "r": 20, "t": 45, "b": 20},
        height=285,
        legend={"orientation": "h", "y": -0.3},
        hovermode="x unified",
    )
    return figure


def forecast_chart(simulation: MineTwin) -> go.Figure:
    history = simulation.prediction_history
    figure = go.Figure(
        go.Scatter(
            x=[prediction.timestamp for prediction in history],
            y=[prediction.hours_to_threshold for prediction in history],
            mode="lines",
            connectgaps=False,
            name="Tiempo hasta umbral",
            line={"color": AMBER, "width": 2},
        )
    )
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Segoe UI, sans-serif", "color": MUTED},
        xaxis_title="Tiempo simulado · UTC",
        yaxis_title="Horas de operación restantes",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=280,
    )
    return figure
