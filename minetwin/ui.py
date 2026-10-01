import streamlit as st

from minetwin.branding import render_brand
from minetwin.paths import PACKAGE_ROOT
from minetwin.scania_ui import render_scania


def main() -> None:
    st.set_page_config(
        page_title="MineTwin · SCANIA Component X",
        page_icon=":material/landscape:",
        layout="wide",
    )
    st.html(PACKAGE_ROOT / "styles.css")
    with st.sidebar:
        render_brand()
    render_scania()
