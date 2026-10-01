import streamlit as st

from minetwin.data.scania import ScaniaReplayView
from minetwin.langflow_client import (
    LangflowClient,
    LangflowConfig,
    LangflowError,
    explanation_context,
)
from minetwin.scania_explanation import ScaniaExplanationSource


@st.cache_resource
def _source() -> ScaniaExplanationSource:
    return ScaniaExplanationSource()


def render_assistant(view: ScaniaReplayView, regime: str) -> None:
    st.subheader("Asistente de Component X · Langflow")
    st.caption(
        "Explica un estado publicado de SCANIA. No modifica la clase, el modelo "
        "ni ejecuta mantenimiento."
    )
    try:
        observation = _source().observation(view, regime)
        context = explanation_context(observation)
    except (OSError, ValueError, LangflowError) as error:
        st.warning(str(error))
        return
    try:
        config = LangflowConfig.from_environment()
    except ValueError as error:
        st.warning(str(error))
        return
    if config is None:
        st.info("Configura la conexión de Langflow para solicitar explicaciones.")
        return
    state = observation.state
    context_key = (
        view.vehicle_id,
        view.split.value,
        view.current.time_step,
        regime,
        state.risk.model_id,
        state.training_id,
        state.data_version,
    )
    st.caption(
        f"Vehículo {view.vehicle_id} · partición {view.node_id} · "
        f"paso {view.current.time_step:g} · clase {state.risk.predicted_class} · "
        f"calidad {state.quality:.1%}"
    )
    with st.expander("Evidencia enviada al flujo", expanded=False):
        st.json(context)
    if st.button("Explicar observación", key="scania_explain"):
        try:
            with st.spinner("Consultando Langflow…"):
                response = LangflowClient(config).explain(observation)
            st.session_state.scania_assistant_response = (context_key, response)
        except LangflowError as error:
            st.error(str(error))
    saved = st.session_state.get("scania_assistant_response")
    if saved and saved[0] == context_key:
        st.write(saved[1])
