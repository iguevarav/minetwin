import csv
import json
from pathlib import Path

from minetwin.demo.payload import CASES_PATH, construir_payload_demo, validar_payload
from minetwin.views import feria
from minetwin.views.feria import MARCADOR, html_con_payload


def test_generated_payload_matches_the_contract():
    assert validar_payload(construir_payload_demo()) == []


def test_example_payload_matches_the_contract():
    example = json.loads(
        Path("minetwin/demo/ejemplo_payload.json").read_text(encoding="utf-8")
    )
    assert validar_payload(example) == []


def test_priority_case_matches_its_demonstration_artifact():
    payload = construir_payload_demo()
    truck = payload["camiones"][0]
    with CASES_PATH.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    row = next(
        item
        for item in rows
        if item["vehicle_id"] == truck["id"]
        and int(item["fold"]) == truck["grupo"]
        and int(item["rank"]) == truck["puesto"]
    )
    assert truck["probabilidades"] == [
        float(row[f"probability_{label}"]) for label in range(5)
    ]
    assert truck["prioritario"] is (row["selected"] == "True")


def test_demo_uses_the_three_declared_priority_positions():
    trucks = construir_payload_demo()["camiones"]
    assert [truck["caso"] for truck in trucks] == [
        "Requiere revisi\u00f3n",
        "Punto medio",
        "No requiere",
    ]
    assert trucks[0]["prioritario"]
    assert not trucks[1]["prioritario"]
    assert not trucks[2]["prioritario"]


def test_federation_payload_uses_the_published_summary():
    federation = construir_payload_demo()["federacion"]
    assert federation == {
        "reportes_enviados": 0,
        "mb_compartidos": 2.96,
        "mb_centralizado": 10.64,
        "nodos": ["Alpha", "Beta", "Gamma"],
    }


def test_threshold_and_history_are_coherent_and_causal():
    truck = construir_payload_demo()["camiones"][0]
    assert truck["prioritario"] is (truck["probabilidades"][4] >= truck["umbral"])
    assert all(reading["t"] <= truck["lecturas"][-1]["t"] for reading in truck["lecturas"])
    assert truck["lecturas"][truck["ventana"]["fin"]]["n"] == truck["lecturas_totales"]


def test_truck_payload_has_no_retrospective_label():
    assert "label" not in json.dumps(construir_payload_demo())


def test_payload_serializes_without_nan_and_replaces_one_marker():
    payload = construir_payload_demo()
    json.dumps(payload, allow_nan=False)
    html = html_con_payload(payload)
    assert html.count(MARCADOR) == 0


def test_demo_uses_the_embedded_example_when_payload_preparation_fails(monkeypatch):
    monkeypatch.setattr(feria, "_html", lambda _: (_ for _ in ()).throw(OSError("x")))
    html, error = feria._prepare_html()
    assert error == "x"
    assert html.count(MARCADOR) == 1
