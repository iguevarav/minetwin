from pathlib import Path

from minetwin.report_ui import prepare_report


def test_report_embeds_local_assets_and_reports_missing(tmp_path: Path):
    report = tmp_path / "report.html"
    report.write_text(
        '<img src="figure.jpg"><a download href="table.xlsx">Excel</a>'
        '<img src="missing.jpg"><img src="../outside.jpg">', encoding="utf-8"
    )
    (tmp_path / "figure.jpg").write_bytes(b"figure")
    (tmp_path / "table.xlsx").write_bytes(b"table")
    rendered, missing = prepare_report(report)
    assert 'src="data:image/jpeg;base64,ZmlndXJl"' in rendered
    assert 'href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,dGFibGU="' in rendered
    assert missing == ["missing.jpg", "../outside.jpg"]
    (tmp_path / "figure.jpg").write_bytes(b"new")
    assert "bmV3" in prepare_report(report)[0]


def test_article_report_is_available_in_main_navigation():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=30)
    app.session_state["scania_navigation"] = "Resultados del artículo"
    app.run()
    assert not app.exception
    assert any(item.value == "Resultados del artículo" for item in app.subheader)
    assert len(app.get("iframe")) == 1
