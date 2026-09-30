import streamlit as st

from minetwin.domain import AssetStatus, ForecastStatus, Intervention, OrderStatus
from minetwin.experiments import export_snapshot
from minetwin.services import MineTwin
from minetwin.visuals import forecast_chart

INTERVENTIONS = {
    Intervention.REPLACEMENT: "Sustitución del componente",
    Intervention.PARTIAL_REPAIR: "Reparación parcial",
}


def component_label(component: str) -> str:
    return {"engine": "Motor", "brakes": "Frenos"}.get(
        component, "Neumático " + component.removeprefix("tire:")
    )


ORDER_STATES = {
    OrderStatus.OPEN: "Abierta",
    OrderStatus.QUEUED: "En cola",
    OrderStatus.IN_PROGRESS: "En progreso",
    OrderStatus.COMPLETED: "Completada",
    OrderStatus.CANCELLED: "Cancelada",
}


def render_forecast(simulation: MineTwin) -> None:
    st.subheader("Pronóstico del motor")
    prediction = simulation.prediction
    if simulation.asset_status != AssetStatus.AVAILABLE:
        st.caption(
            "Pronóstico suspendido durante la parada. Se reevalúa al volver a operar."
        )
        return
    if prediction is None:
        st.caption("Se necesitan observaciones para estimar una tendencia.")
        return
    if prediction.status == ForecastStatus.ESTIMABLE:
        st.metric(
            "Tiempo estimado hasta umbral", f"{prediction.hours_to_threshold:.2f} h"
        )
    elif prediction.status == ForecastStatus.OUTSIDE_HORIZON:
        st.info("Fuera del horizonte útil", icon=":material/schedule:")
    else:
        st.caption("No estimable · " + prediction.reason)
    st.caption(f"Horas de operación · Observación {prediction.timestamp:%H:%M:%S} UTC")
    with st.expander("Cómo se estima", icon=":material/analytics:"):
        st.write(prediction.reason)
        st.caption(
            "La tendencia combina desviaciones de temperatura y vibración por régimen. "
            "Es una extrapolación del modelo simulado; no una RUL industrial validada."
        )
        if prediction.indicator is not None:
            st.write(
                f"Indicador: {prediction.indicator:.3f} · Umbral: {simulation.predictor_config.critical_indicator:.3f}"
            )
        if prediction.slope_per_hour is not None:
            st.write(
                f"Pendiente: {prediction.slope_per_hour:.3f}/h · R²: {prediction.r_squared:.3f}"
            )
        st.caption(
            f"{prediction.sample_count} muestras operativas · {prediction.rules_version}"
        )


def render_maintenance_controls(simulation: MineTwin) -> None:
    order = simulation.current_order
    with st.container(border=True):
        st.subheader("Intervenir el activo")
        component = st.selectbox(
            "Componente",
            simulation.profile.components,
            format_func=component_label,
            key=f"component_{simulation.truck_id}",
            disabled=order is not None,
        )
        if order:
            st.caption("Componente de la orden: " + component_label(order.component))
        intervention = st.selectbox(
            "Intervención",
            list(INTERVENTIONS),
            format_func=INTERVENTIONS.get,
            key="intervention",
            disabled=order is not None,
        )
        controls = st.columns(3)
        diagnoses = (
            (simulation.twin.diagnosis, *simulation.twin.component_diagnoses)
            if simulation.twin
            else ()
        )
        reason = next(
            (
                diagnosis.explanation
                for diagnosis in diagnoses
                if diagnosis.component == component
            ),
            "Intervención manual.",
        )
        controls[0].button(
            "Crear orden",
            key="create_order",
            icon=":material/add:",
            disabled=order is not None or simulation.twin is None,
            on_click=simulation.create_order,
            args=(intervention, reason, component),
            width="stretch",
        )
        controls[1].button(
            "Iniciar intervención",
            key="start_order",
            type="primary",
            icon=":material/build:",
            disabled=order is None or order.status != OrderStatus.OPEN,
            on_click=simulation.start_order,
            args=(order.id if order else "",),
            width="stretch",
        )
        controls[2].button(
            "Cancelar orden",
            key="cancel_order",
            icon=":material/close:",
            disabled=order is None or order.status != OrderStatus.OPEN,
            on_click=simulation.cancel_order,
            args=(order.id if order else "",),
            width="stretch",
        )
        st.caption(
            "La reparación termina al cumplir su duración simulada. Usa los controles de avance."
        )


def render_maintenance_status(simulation: MineTwin) -> None:
    order = simulation.current_order
    if order:
        st.subheader(f"{order.id} · {ORDER_STATES[order.status]}")
        if (
            order.status == OrderStatus.IN_PROGRESS
            and order.due_at
            and order.duration_seconds
        ):
            remaining = max(0.0, (order.due_at - simulation.time).total_seconds())
            st.progress(1 - remaining / order.duration_seconds)
            kind = "Correctiva" if order.corrective else "Preventiva"
            st.caption(
                f"{kind} · Restan {remaining / 60:.1f} min simulados · Camión detenido"
            )
        else:
            st.caption(
                "Orden preparada. La parada comienza al iniciar la intervención."
            )
    elif simulation.asset_status == AssetStatus.FAILED:
        st.error("Camión detenido por falla. Crea e inicia una orden correctiva.")
    else:
        st.caption("Sin intervención en curso.")
    metrics = simulation.metrics
    columns = st.columns(4)
    availability = metrics["availability"]
    columns[0].metric(
        "Disponibilidad", "Sin datos" if availability is None else f"{availability:.1%}"
    )
    columns[1].metric("Fallas", metrics["failures"])
    columns[2].metric("Intervenciones", metrics["interventions"])
    columns[3].metric(
        "Tiempo detenido",
        f"{(metrics['failed_seconds'] + metrics['maintenance_seconds']) / 60:.1f} min",
    )
    render_forecast(simulation)
    if simulation.prediction_history:
        with st.expander("Evolución del pronóstico", icon=":material/show_chart:"):
            st.plotly_chart(
                forecast_chart(simulation),
                theme=None,
                width="stretch",
                key="forecast_chart",
            )
    if simulation.orders:
        st.subheader("Órdenes registradas")
        st.dataframe(
            [
                {
                    "Orden": item.id,
                    "Intervención": INTERVENTIONS[item.intervention],
                    "Componente": component_label(item.component),
                    "Estado": ORDER_STATES[item.status],
                    "Inicio (UTC)": item.started_at,
                    "Cierre (UTC)": item.closed_at,
                    "Duración (min)": item.duration_seconds / 60
                    if item.duration_seconds
                    else None,
                }
                for item in reversed(simulation.orders)
            ],
            hide_index=True,
            width="stretch",
        )


def render_export(simulation: MineTwin) -> None:
    if simulation.twin:
        with st.expander("Exportar evidencia", icon=":material/download:"):
            st.download_button(
                "Descargar historial disponible",
                data=export_snapshot(simulation),
                file_name="minetwin_historial.zip",
                mime="application/zip",
                on_click="ignore",
            )
            st.caption(
                "Incluye observaciones, pronósticos, eventos, órdenes y configuración retenidos en esta sesión."
            )
