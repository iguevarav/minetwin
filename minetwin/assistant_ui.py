import streamlit as st

from minetwin.federation import FederatedFleet
from minetwin.langflow_client import LangflowClient, LangflowConfig, LangflowError


def render_assistant(fleet: FederatedFleet, truck_id: str) -> None:
    with st.expander(
        "Asistente de explicación · Langflow", icon=":material/psychology:"
    ):
        try:
            config = LangflowConfig.from_environment()
        except ValueError:
            st.caption("La conexión de Langflow necesita una configuración válida.")
            return
        if config is None:
            st.caption(
                "Conexión pendiente. Las variables necesarias están indicadas en el plan del proyecto."
            )
            return
        view = fleet.view(truck_id)
        context_key = (
            truck_id,
            view.twin.version if view.twin else None,
            view.asset_status,
            view.orders,
        )
        st.caption(
            "Envía la observación normalizada y sus diagnósticos al flujo configurado. La respuesta es orientativa y no ejecuta mantenimiento."
        )
        if st.button(
            "Explicar condición actual",
            key="explain_condition",
            disabled=view.stale or view.twin is None or fleet.running,
        ):
            try:
                with st.spinner("Preparando explicación…"):
                    response = LangflowClient(config).explain(view)
                st.session_state.assistant_response = (context_key, response)
            except LangflowError as error:
                st.error(str(error))
        if fleet.running:
            st.caption("Pausa la simulación para analizar una observación concreta.")
        saved = st.session_state.get("assistant_response")
        if (
            saved
            and view.twin
            and saved[0] == context_key
            and not view.stale
            and not fleet.running
        ):
            st.text(saved[1])
