from pathlib import Path

from streamlit.testing.v1 import AppTest

from minetwin.domain import OrderStatus

APP_PATH = Path(__file__).parents[1] / "simulation_app.py"


def _simulation_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=20)


def test_truck_selection_federation_quality_disconnect_and_reconnect():
    app = _simulation_app().run()
    app.button(key="step").click().run()
    initial_time = app.session_state.fleet.time
    app.selectbox(key="selected_truck").set_value("TRUCK-004").run()
    assert not app.exception
    assert app.session_state.simulation.truck_id == "TRUCK-004"
    assert app.session_state.fleet.time == initial_time
    app.radio(key="navigation").set_value("Federación").run()
    assert not app.exception
    assert app.code[0].language == "xml"
    app.selectbox(key="federated_node").set_value("beta").run()
    app.button(key="bad_packet").click().run()
    assert app.session_state.fleet.quality("beta").rejected == 1
    app.button(key="duplicate_packet").click().run()
    assert app.session_state.fleet.quality("beta").duplicates == 1
    app.button(key="toggle_node").click().run()
    assert not app.exception
    assert app.warning
    app.radio(key="navigation").set_value("Mantenimiento").run()
    assert not app.exception
    assert "create_order" not in [button.key for button in app.button]
    app.button(key="step").click().run()
    assert app.session_state.fleet.view("TRUCK-004").age_seconds == 60
    app.radio(key="navigation").set_value("Federación").run()
    app.button(key="toggle_node").click().run()
    assert not app.exception
    assert app.session_state.simulation.twin.version == 2


def test_maintenance_targets_selected_component_and_truck():
    app = _simulation_app().run()
    app.button(key="step").click().run()
    app.selectbox(key="selected_truck").set_value("TRUCK-009").run()
    app.radio(key="navigation").set_value("Mantenimiento").run()
    app.selectbox(key="component_TRUCK-009").set_value("tire:FL").run()
    app.button(key="create_order").click().run()
    order = app.session_state.simulation.current_order
    assert order.component == "tire:FL"
    assert order.truck_id == "TRUCK-009"
    app.button(key="start_order").click().run()
    assert app.session_state.simulation.current_order.status == OrderStatus.IN_PROGRESS
    assert not app.session_state.fleet.local("TRUCK-001").orders
    assert not app.exception


