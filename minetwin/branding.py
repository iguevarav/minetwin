import streamlit as st


def render_brand() -> None:
    st.html(
        '<div class="mt-brand"><div class="mt-brand-mark">'
        '<svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">'
        '<path d="M3 22L11 6L16 15L20 9L26 22Z" fill="none" '
        'stroke="currentColor" stroke-width="2.3" stroke-linejoin="round"/>'
        '</svg></div><div><div class="mt-brand-name">Mine<span>Twin</span></div>'
        '<div class="mt-eyebrow">INTELIGENCIA OPERATIVA</div></div></div>'
    )
