from dataclasses import asdict

import streamlit as st

from minetwin.domain import EngineStatus
from minetwin.federation import FederatedFleet, TruckView


def render_node_controls(fleet: FederatedFleet) -> None:
    nodes = list(fleet.nodes)
    selected = st.session_state.get("last_selected_node", nodes[0])
    node_id = st.selectbox(
        "Nodo", nodes, index=nodes.index(selected), key="federated_node"
    )
    st.session_state.last_selected_node = node_id
    node = fleet.nodes[node_id]
    columns = st.columns(3)
    columns[0].button(
        "Desconectar" if node.connected else "Reconectar",
        key="toggle_node",
        on_click=fleet.set_connected,
        args=(node_id, not node.connected),
        icon=":material/settings_ethernet:",
        width="stretch",
    )
    columns[1].button(
        "Probar paquete inválido",
        key="bad_packet",
        disabled=not node.connected,
        on_click=fleet.demonstrate_bad_packet,
        args=(node_id,),
        width="stretch",
    )
    columns[2].button(
        "Repetir último paquete",
        key="duplicate_packet",
        disabled=not node.connected,
        on_click=fleet.demonstrate_bad_packet,
        args=(node_id, True),
        width="stretch",
    )


def render_fleet_summary(fleet: FederatedFleet) -> None:
    with st.expander("Flota · 9 camiones / 3 nodos", icon=":material/local_shipping:"):
        rows = []
        for truck_id in fleet.truck_ids:
            view = fleet.view(truck_id)
            statuses = {
                "normal": "Sin anomalías",
                "watch": "En observación",
                "alert": "Alerta",
                "unknown": "No estimable",
            }
            rows.append(
                {
                    "Camión": truck_id,
                    "Nodo": view.node_id,
                    "Perfil": view.profile.id,
                    "Capacidad (t)": view.profile.capacity_tonnes,
                    "Datos": "Desactualizados"
                    if view.stale
                    else ("Vigentes" if view.twin else "Sin lecturas"),
                    "Captura (UTC)": view.twin.observation.source_time
                    if view.twin
                    else None,
                    "Antigüedad (s)": view.age_seconds,
                    "Motor": statuses[view.twin.diagnosis.status]
                    if view.twin
                    else "Sin lecturas",
                    "Alertas abiertas": sum(
                        alert.closed_at is None for alert in view.alerts
                    ),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")


def render_stale_view(view: TruckView) -> None:
    st.warning(
        "Nodo desconectado o datos caducados. Se conserva la última observación.",
        icon=":material/cloud_off:",
    )
    if view.twin:
        st.caption(
            f"Última captura: {view.twin.observation.source_time:%d/%m/%Y %H:%M:%S} UTC · Antigüedad {view.age_seconds / 60:.1f} min"
        )
        st.write("Último diagnóstico registrado: " + view.twin.diagnosis.explanation)
    st.caption(
        "El pronóstico no está vigente. Se necesita comunicación y una observación reciente para reanudar esta vista."
    )


def render_components(simulation) -> None:
    if simulation.twin is None:
        return
    statuses = {
        EngineStatus.NORMAL: "Sin anomalías",
        EngineStatus.WATCH: "En observación",
        EngineStatus.ALERT: "Alerta",
        EngineStatus.UNKNOWN: "No estimable",
    }
    with st.expander("Frenos y neumáticos", icon=":material/tire_repair:"):
        st.caption(
            "Diagnóstico por componente. Sin modelo de RUL para frenos o neumáticos."
        )
        for diagnosis in simulation.twin.component_diagnoses:
            label = (
                "Frenos"
                if diagnosis.component == "brakes"
                else "Neumático " + diagnosis.component.split(":")[1]
            )
            text = f"{label} · {statuses[diagnosis.status]}"
            if diagnosis.status == EngineStatus.ALERT:
                st.error(text)
            elif diagnosis.status == EngineStatus.WATCH:
                st.warning(text)
            else:
                st.write(text)
            st.caption(diagnosis.explanation)


def render_federation(fleet: FederatedFleet, truck_id: str) -> None:
    rows = []
    for node in fleet.nodes.values():
        quality = fleet.quality(node.id)
        rows.append(
            {
                "Nodo": node.id,
                "Formato": node.wire_format.value.upper(),
                "Comunicación": "Conectado" if node.connected else "Desconectado",
                "Completitud": f"{quality.present}/{quality.required}"
                if quality.required
                else "Sin datos",
                "Validez": f"{quality.valid}/{quality.present}"
                if quality.present
                else "Sin datos",
                "Aceptados": quality.accepted,
                "Rechazados": quality.rejected,
                "Duplicados": quality.duplicates,
                "Tardíos": quality.late,
                "Contradicciones": quality.inconsistent,
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Los indicadores desconectados conservan su último valor publicado. Los denominadores incluyen campos requeridos; no se sustituyen ausentes por cero."
    )
    view = fleet.view(truck_id)
    if view.stale:
        render_stale_view(view)
        return
    simulation = fleet.access(truck_id)
    if simulation.twin:
        st.subheader(f"Trazabilidad · {truck_id}")
        source, canonical = st.columns(2)
        with source:
            st.caption("Dato de origen · " + simulation.wire_format.value.upper())
            st.code(simulation.raw_history[-1], language=simulation.wire_format.value)
        with canonical:
            st.caption("Contrato común")
            st.json(asdict(simulation.twin.observation), expanded=False)
    issues = fleet.issues(view.node_id)
    if issues:
        with st.expander("Últimos rechazos y eventos de calidad"):
            for issue in issues[-5:]:
                st.text(issue)
