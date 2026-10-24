from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_main_application_uses_only_scania():
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not app.exception
    assert not app.warning
    assert len(app.button) == 1
    assert app.button[0].label.endswith("Salir del modo feria")
    app.button[0].click().run()
    assert not app.exception
    assert app.title[0].value.startswith("MineTwin")
    assert app.title[0].value.endswith("Component X")


def test_every_view_offers_one_download_per_table_and_chart():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    # All demo sections read the same payload as the animated presentation.
    choices = app.selectbox(key="demo_export_section").options
    for section in choices:
        app.selectbox(key="demo_export_section").set_value(section).run()
        assert not app.exception
        _assert_download_coverage(app)
    app.button[0].click().run()
    for view in (
        "Dashboard",
        "Datos y EDA",
        "Telemetría",
        "Riesgo predictivo",
        "Aprendizaje",
        "Federación",
    ):
        app.radio(key="scania_navigation").set_value(view).run()
        assert not app.exception, view
        _assert_download_coverage(app)


def _assert_download_coverage(app):
    count = len(app.dataframe) + len(app.get("plotly_chart"))
    assert count > 0
    downloads = app.get("download_button")
    assert len(downloads) == count
    assert all("Excel" in item.label or "600 DPI" in item.label for item in downloads)
