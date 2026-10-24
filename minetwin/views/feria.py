import json

import streamlit as st
import streamlit.components.v1 as components

from minetwin.demo.payload import construir_payload_demo
from minetwin.learning.provenance import file_sha256
from minetwin.paths import PACKAGE_ROOT, result_path
from minetwin.views.demo_exports import render_demo_exports

PLANTILLA = PACKAGE_ROOT / "demo" / "flujo.html"
MARCADOR = "/*__MINETWIN_PAYLOAD__*/null"
MANIFEST_PATH = result_path("11_mejora", "07_demostracion", "manifest.json")
POLICY_PATH = result_path("11_mejora", "05_politica_prioridad_v2", "policy.json")


def render_feria() -> None:
    st.html(_FULL_SCREEN_CSS)
    st.button(
        "← Salir del modo feria",
        key="exit_fair_mode",
        on_click=_exit_fair_mode,
    )
    html, error = _prepare_html()
    if error is not None:
        st.warning(
            "No se pudo preparar la demostracion real: "
            f"{error}. Se muestra un ejemplo: los camiones 2 y 3 son ilustrativos."
        )
    payload = (
        _payload(_artifact_signature())
        if error is None
        else json.loads(
            (PLANTILLA.parent / "ejemplo_payload.json").read_text(encoding="utf-8")
        )
    )
    render_demo_exports(payload)
    components.html(html, height=860, scrolling=False)


def html_con_payload(payload: dict | None) -> str:
    template = PLANTILLA.read_text(encoding="utf-8")
    if payload is None:
        return template
    if template.count(MARCADOR) != 1:
        raise ValueError("La plantilla de feria no contiene un marcador único.")
    value = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace(
        "</", "<\\/"
    )
    return template.replace(MARCADOR, value, 1)


@st.cache_data(show_spinner="Preparando la demo...")
def _html(signature: str) -> str:
    return html_con_payload(_payload(signature))


@st.cache_data(show_spinner=False)
def _payload(signature: str) -> dict:
    return construir_payload_demo()


def _prepare_html() -> tuple[str, str | None]:
    try:
        return _html(_artifact_signature()), None
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        return html_con_payload(None), str(error)


def _artifact_signature() -> str:
    return ":".join((file_sha256(MANIFEST_PATH), file_sha256(POLICY_PATH)))


def _exit_fair_mode() -> None:
    st.session_state.scania_navigation = "Dashboard"


_FULL_SCREEN_CSS = """
<style>
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
.stDeployButton {
    display: none;
}
[data-testid="stMainBlockContainer"] {
    max-width: none;
    padding: 0.25rem 0.5rem 0 !important;
}
</style>
"""
