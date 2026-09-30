from pathlib import Path

import streamlit as st

from minetwin.assistant_ui import render_assistant
from minetwin.domain import WHEELS, AssetStatus, EngineStatus, OperatingState, Scenario
from minetwin.federation import FederatedFleet
from minetwin.federation_ui import (
    render_components,
    render_federation,
    render_fleet_summary,
    render_node_controls,
    render_stale_view,
)
from minetwin.maintenance_ui import (
    component_label,
    render_export,
    render_forecast,
    render_maintenance_controls,
    render_maintenance_status,
)
from minetwin.research_ui import render_workshop_results
from minetwin.scania_ui import render_scania
from minetwin.services import MineTwin
from minetwin.simulation import SimulationConfig
from minetwin.visuals import operation_map, signal_chart

SCENARIOS = {
    Scenario.NORMAL: "Operación normal",
    Scenario.ENGINE_DEGRADATION: "Degradación progresiva del motor",
    Scenario.ENGINE_VARIABLE_DEGRADATION: "Degradación variable por régimen",
    Scenario.BRAKE_STRESS: "Sobretemperatura de frenos",
    Scenario.TIRE_LEAK: "Pérdida de presión en neumático FL",
}
OPERATING_STATES = {
    OperatingState.WAITING: "Espera",
    OperatingState.LOADING: "Carga",
    OperatingState.HAULING: "Traslado cargado",
    OperatingState.UNLOADING: "Descarga",
    OperatingState.RETURNING: "Retorno vacío",
}
ENGINE_STATES = {
    EngineStatus.NORMAL: "Motor sin anomalías detectadas",
    EngineStatus.WATCH: "Motor en observación",
    EngineStatus.ALERT: "Alerta de motor",
    EngineStatus.UNKNOWN: "Motor no estimable",
}
SIGNALS = {
    "engine_temperature": ("Temperatura del motor", "°C"),
    "engine_vibration": ("Vibración del motor", "mm/s"),
    "load_ratio": ("Carga relativa", "1"),
    "operating_hours": ("Horómetro", "h"),
    "brake_temperature": ("Temperatura de frenos", "°C"),
    "slope_percent": ("Pendiente", "%"),
    **{f"tire_pressure_{wheel}": (f"Presión {wheel}", "kPa") for wheel in WHEELS},
}
VIEWS = {
    "Resumen": "Centro de operaciones",
    "Telemetría": "Condición del activo",
    "Alertas": "Seguimiento de alertas",
    "Mantenimiento": "Mantenimiento del activo",
    "Federación": "Nodos y calidad de datos",
    "Coordinación": "Taller y coordinación de flota",
}
RESEARCH_VIEWS = {"Coordinación"}
APPLICATION_MODES = {
    "scania": "Datos reales · SCANIA",
    "simulation": "Experimento simulado",
}


def _render_application_mode() -> str:
    st.html(
        '<div class="mt-brand"><div class="mt-brand-mark">'
        '<svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">'
        '<path d="M3 22L11 6L16 15L20 9L26 22Z" fill="none" '
        'stroke="currentColor" stroke-width="2.3" stroke-linejoin="round"/>'
        '</svg></div><div><div class="mt-brand-name">Mine<span>Twin</span></div>'
        '<div class="mt-eyebrow">INTELIGENCIA OPERATIVA</div></div></div>'
        '<div class="mt-eyebrow mt-section-label">FUENTE DE DATOS</div>'
    )
    return st.radio(
        "Fuente de datos",
        tuple(APPLICATION_MODES),
        format_func=APPLICATION_MODES.get,
        label_visibility="collapsed",
        key="application_mode",
    )


def _render_sidebar(fleet: FederatedFleet) -> tuple[str, list[str], int]:
    st.html('<div class="mt-eyebrow mt-section-label">EXPERIMENTO</div>')
    view = st.radio(
        "Navegación", list(VIEWS), label_visibility="collapsed", key="navigation"
    )
    st.divider()
    if view in RESEARCH_VIEWS:
        st.caption("EVIDENCIA DE INVESTIGACIÓN")
        st.caption("SCANIA Component X · Evaluación reproducible")
        return view, [], 1
    st.caption("ACTIVO SIMULADO")
    truck_id = st.selectbox("Camión", fleet.truck_ids, key="selected_truck")
    profile = fleet.view(truck_id).profile
    st.caption(f"{profile.id} · {profile.capacity_tonnes:g} t")
    with st.expander("Escenario", icon=":material/tune:"):
        with st.form("scenario_form"):
            scenario = st.selectbox(
                "Operación", list(SCENARIOS), format_func=SCENARIOS.get, key="scenario"
            )
            seed = st.number_input(
                "Semilla", min_value=0, max_value=2**32 - 1, value=42, step=1
            )
            if st.form_submit_button("Reiniciar escenario", width="stretch"):
                st.session_state.fleet = FederatedFleet(
                    SimulationConfig(seed=int(seed), scenario=scenario)
                )
                st.session_state.pop("assistant_response", None)
        st.caption("Reiniciar aplica los cambios y borra el historial.")
    with st.expander("Visualización", icon=":material/show_chart:"):
        selected_signals = st.multiselect(
            "Señales visibles",
            list(SIGNALS),
            default=["engine_temperature", "engine_vibration"],
            format_func=lambda signal: SIGNALS[signal][0],
            key="visible_signals",
        )
        speed = st.select_slider("Pasos por actualización", options=[1, 5, 10], value=1)
    st.caption("Simulación académica · Datos sintéticos")
    return view, selected_signals, speed


def _render_controls(simulation: MineTwin) -> None:
    with st.container(key="controls"):
        controls = st.columns(3)
        controls[0].button(
            "Iniciar",
            on_click=simulation.start,
            disabled=simulation.running,
            key="start",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
        )
        controls[1].button(
            "Pausar",
            on_click=simulation.pause,
            disabled=not simulation.running,
            key="pause",
            icon=":material/pause:",
            width="stretch",
        )
        controls[2].button(
            "Avanzar un paso",
            on_click=simulation.advance,
            disabled=simulation.running,
            key="step",
            icon=":material/skip_next:",
            width="stretch",
        )


def _render_metrics(simulation: MineTwin) -> None:
    twin = simulation.twin
    with st.container(key="kpis"):
        columns = st.columns(4)
        for column, (label, signal, unit) in zip(
            columns,
            (
                ("Temperatura motor", "engine_temperature", "°C"),
                ("Vibración motor", "engine_vibration", "mm/s"),
                ("Carga útil", "load_ratio", "t"),
                ("Horómetro", "operating_hours", "h"),
            ),
            strict=True,
        ):
            value = twin.observation.reading(signal).value if twin else None
            if signal == "load_ratio" and value is not None:
                value *= simulation.profile.capacity_tonnes
            column.metric(
                label, "Sin datos" if value is None else f"{value:.2f} {unit}"
            )


def _render_alert_status(simulation: MineTwin) -> None:
    if simulation.asset_status == AssetStatus.MAINTENANCE:
        st.info("Activo en mantenimiento", icon=":material/build:")
        st.caption("La operación y el horómetro están detenidos.")
        return
    if simulation.asset_status == AssetStatus.FAILED:
        st.error(
            "Falla de motor · Requiere reparación correctiva", icon=":material/error:"
        )
        return
    twin = simulation.twin
    if twin is None:
        st.info("Sin observaciones", icon=":material/sensors_off:")
        st.caption("Avanza un paso o inicia la simulación para observar el motor.")
        return
    status = twin.diagnosis.status
    with st.container(key=f"status_{status.value}"):
        if status == EngineStatus.ALERT:
            st.error(ENGINE_STATES[status], icon=":material/error:")
            st.caption("Desviación persistente. Se recomienda inspeccionar el motor.")
        elif status == EngineStatus.WATCH:
            st.warning(ENGINE_STATES[status], icon=":material/warning:")
            st.caption("La desviación aún no cumple la regla de persistencia.")
        elif status == EngineStatus.UNKNOWN:
            st.info(ENGINE_STATES[status], icon=":material/sensors_off:")
            st.caption("Se necesitan lecturas válidas para evaluar su condición.")
        else:
            st.success(ENGINE_STATES[status], icon=":material/check_circle:")
            st.caption("Las señales están dentro de las referencias del régimen.")
    engine_alert = next(
        (
            alert
            for alert in simulation.alerts
            if alert.component == "engine" and alert.closed_at is None
        ),
        None,
    )
    if engine_alert:
        st.caption(f"Alerta abierta desde {engine_alert.opened_at:%H:%M} UTC")
    with st.expander("Ver diagnóstico", icon=":material/analytics:"):
        st.write(twin.diagnosis.explanation)
        st.caption(
            f"Regla {twin.diagnosis.rule_id} · Versión {twin.diagnosis.rules_version}"
        )


def _render_map(simulation: MineTwin) -> None:
    st.subheader("Mapa de operación")
    st.caption("Ciclo de acarreo · Esquema referencial, sin coordenadas reales")
    st.html(operation_map(simulation))
    twin = simulation.twin
    state = (
        OPERATING_STATES[twin.observation.operating_state] if twin else "Sin lecturas"
    )
    if simulation.asset_status != AssetStatus.AVAILABLE:
        state = (
            "En mantenimiento"
            if simulation.asset_status == AssetStatus.MAINTENANCE
            else "Detenido por falla"
        )
    cycles = str(twin.observation.cycles) if twin else "—"
    st.html(
        f'<div class="mt-map-footer"><span>{simulation.truck_id} · {state}</span>'
        f"<span>{cycles} ciclos completados</span></div>"
    )


def _render_charts(simulation: MineTwin, selected_signals: list[str]) -> None:
    if not simulation.twin:
        return
    if not selected_signals:
        st.caption("Selecciona señales en Visualización para consultar su evolución.")
        return
    with st.container(key="charts"):
        for start in range(0, len(selected_signals), 2):
            columns = st.columns(2)
            for column, signal in zip(columns, selected_signals[start : start + 2]):
                with column:
                    label, unit = SIGNALS[signal]
                    st.plotly_chart(
                        signal_chart(simulation, signal, label, unit),
                        width="stretch",
                        key=f"chart_{signal}",
                        theme=None,
                        config={"displayModeBar": False, "responsive": True},
                    )
    st.caption(
        "Línea discontinua: umbral. Roja: alerta. Verde: intervención completada."
    )


def _render_observation_details(simulation: MineTwin) -> None:
    if simulation.twin is None:
        return
    packet = simulation.twin.observation
    with st.expander("Lecturas y calidad", icon=":material/fact_check:"):
        st.dataframe(
            [
                {
                    "Señal": SIGNALS[reading.name][0],
                    "Valor": reading.value,
                    "Unidad": reading.unit,
                    "Calidad": {
                        "valid": "Válida",
                        "missing": "Ausente",
                        "invalid": "Inválida",
                    }[reading.quality],
                    "Motivo": reading.reason or "",
                }
                for reading in packet.readings
            ],
            hide_index=True,
            width="stretch",
        )
    with st.expander("Trazabilidad de la observación", icon=":material/data_object:"):
        st.caption(
            f"{packet.event_id} · Nodo {packet.node_id} · Esquema {packet.schema_version}"
        )
        st.code(simulation.raw_history[-1], language=simulation.wire_format.value)


def _render_alert_history(simulation: MineTwin) -> None:
    st.subheader("Historial de alertas")
    if not simulation.alerts:
        st.caption("No hay alertas registradas en este escenario.")
        return
    st.dataframe(
        [
            {
                "Activo": alert.truck_id,
                "Componente": component_label(alert.component),
                "Estado": "Cerrada" if alert.closed_at else "Abierta",
                "Apertura (UTC)": alert.opened_at,
                "Cierre (UTC)": alert.closed_at,
            }
            for alert in reversed(simulation.alerts)
        ],
        hide_index=True,
        width="stretch",
    )


def _render_view(simulation: MineTwin, view: str, selected_signals: list[str]) -> None:
    if view != "Mantenimiento":
        _render_metrics(simulation)
    if simulation.twin:
        st.caption(
            f"Última observación · {simulation.twin.observation.source_time:%d/%m/%Y · %H:%M:%S} UTC"
        )
    if view == "Resumen":
        with st.container(key="overview"):
            map_column, alert_column = st.columns([1.7, 1], gap="large")
            with map_column, st.container(border=True, key="map_panel"):
                _render_map(simulation)
            with alert_column, st.container(border=True, key="alert_panel"):
                st.subheader("Condición y alertas")
                _render_alert_status(simulation)
                st.divider()
                st.caption(
                    f"{simulation.alert_count} alertas generadas · Todos los componentes"
                )
                render_forecast(simulation)
        st.subheader("Señales principales")
        _render_charts(simulation, selected_signals[:2])
    elif view == "Telemetría":
        _render_charts(simulation, selected_signals)
        _render_observation_details(simulation)
        if not simulation.twin:
            st.info("Inicia la simulación para consultar las señales.")
    elif view == "Alertas":
        _render_alert_status(simulation)
        _render_alert_history(simulation)
    else:
        render_maintenance_status(simulation)
    if view in ("Resumen", "Alertas", "Telemetría"):
        render_components(simulation)


def main() -> None:
    st.set_page_config(
        page_title="MineTwin · Operaciones",
        page_icon=":material/landscape:",
        layout="wide",
    )
    st.html(Path(__file__).with_name("styles.css"))
    with st.sidebar:
        application_mode = _render_application_mode()
    if application_mode == "scania":
        render_scania()
        return
    if "fleet" not in st.session_state:
        initial = st.session_state.get("simulation")
        st.session_state.fleet = FederatedFleet(
            first_truck=initial if isinstance(initial, MineTwin) else None
        )
    with st.sidebar:
        view, selected_signals, speed = _render_sidebar(st.session_state.fleet)
    if view in RESEARCH_VIEWS:
        st.html('<div class="mt-eyebrow">INVESTIGACIÓN / RESULTADOS</div>')
        st.title(VIEWS[view])
        render_workshop_results()
        return
    fleet = st.session_state.fleet
    truck_id = st.session_state.selected_truck
    simulation = fleet.access(truck_id)
    st.session_state.simulation = simulation
    snapshot = fleet.view(truck_id)
    st.html(
        f'<div class="mt-eyebrow">OPERACIÓN MINERA / NODO {snapshot.node_id.upper()}</div>'
    )
    st.title(VIEWS[view])
    st.caption(
        f"{truck_id} · {SCENARIOS[fleet.config.scenario]} · Semilla de flota {fleet.config.seed}"
    )
    _render_controls(fleet)
    badge_class = "active" if fleet.running else ""
    badge_label = "Simulación en marcha" if fleet.running else "Simulación en pausa"
    st.html(f'<span class="mt-badge {badge_class}">{badge_label}</span>')
    if view == "Federación":
        render_node_controls(fleet)
    if view == "Mantenimiento" and not snapshot.stale:
        render_maintenance_controls(simulation)
    revision = st.session_state.get("render_revision", 0) + 1
    st.session_state.render_revision = revision

    @st.fragment(run_every="1s" if fleet.running else None)
    def live_observation() -> None:
        previous_revision = st.session_state.get("fragment_revision")
        st.session_state.fragment_revision = revision
        is_periodic_update = previous_revision == revision
        if fleet.running and is_periodic_update:
            previous = fleet.view(truck_id)
            fleet.advance(speed)
            current = fleet.view(truck_id)
            if (previous.asset_status, previous.orders, previous.stale) != (
                current.asset_status,
                current.orders,
                current.stale,
            ):
                st.rerun()
        current = fleet.view(truck_id)
        if view == "Federación":
            render_federation(fleet, truck_id)
        elif current.stale:
            render_stale_view(current)
        else:
            _render_view(simulation, view, selected_signals)
            if view == "Mantenimiento":
                render_export(simulation)
        if view == "Resumen":
            render_fleet_summary(fleet)

    live_observation()
    if view in ("Resumen", "Alertas", "Mantenimiento"):
        render_assistant(fleet, truck_id)
