"""Presenta el informe generado sin mantener una copia desactualizada."""

import base64
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import streamlit as st
import streamlit.components.v1 as components

from minetwin.paths import RESULTS_ROOT

_ASSET = re.compile(r'''\b(src|href)\s*=\s*(["'])(.*?)\2''', re.IGNORECASE)


def prepare_report(path: Path) -> tuple[str, list[str]]:
    """Incrusta los recursos locales para que funcionen dentro de Streamlit."""
    html = path.read_text(encoding="utf-8")
    missing: list[str] = []
    encoded: dict[Path, str] = {}
    root = path.parent.resolve()

    def replace(match: re.Match[str]) -> str:
        attribute, quote, value = match.groups()
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path or value.startswith("#"):
            return match.group(0)
        asset = (root / unquote(parsed.path)).resolve()
        if not asset.is_relative_to(root) or not asset.is_file():
            missing.append(value)
            return f'{attribute}={quote}{quote}'
        if asset not in encoded:
            mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
            data = base64.b64encode(asset.read_bytes()).decode("ascii")
            encoded[asset] = f"data:{mime};base64,{data}"
        return f'{attribute}={quote}{encoded[asset]}{quote}'

    return _ASSET.sub(replace, html), missing


def render_article_report(root: Path = RESULTS_ROOT) -> None:
    path = root / "10_reporte" / "articulo" / "report.html"
    st.subheader("Resultados del artículo")
    st.caption(
        "Informe reproducible con tablas, figuras y sus interpretaciones. "
        "Seleccione las descargas dentro del informe para obtener los archivos originales."
    )
    if not path.is_file():
        st.info(
            "El informe aún no está disponible. Genérelo con "
            "`python -m minetwin.learning.article_report --results results "
            "--output results/10_reporte/articulo --overwrite`."
        )
        return
    try:
        report, missing = prepare_report(path)
    except (OSError, UnicodeError, ValueError) as error:
        st.error(f"No se pudo abrir el informe: {error}")
        return
    if missing:
        st.warning("Faltan recursos del informe: " + ", ".join(sorted(set(missing))))
    components.html(report, height=850, scrolling=True)
