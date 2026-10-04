from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_main_application_uses_only_scania():
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value == "MineTwin · Análisis histórico de Component X"
