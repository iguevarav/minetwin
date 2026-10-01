from pathlib import Path

from streamlit.testing.v1 import AppTest

from minetwin import MineTwin, SimulationConfig
from minetwin.domain import AssetStatus, OrderStatus, Scenario
from minetwin.maintenance import MaintenanceConfig

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SIMULATION_APP_PATH = APP_PATH.with_name("simulation_app.py")


def _simulation_app(timeout: int = 15) -> AppTest:
    return AppTest.from_file(SIMULATION_APP_PATH, default_timeout=timeout)


def test_main_application_uses_only_scania():
    app = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not app.exception
    assert "application_mode" not in {radio.key for radio in app.radio}
    assert app.title[0].value == "MineTwin · Análisis histórico de Component X"
    assert not {"start", "step", "pause"} & {button.key for button in app.button}


def test_navigation_and_filters_do_not_advance_simulation():
    app = _simulation_app().run()
    assert not app.exception
    assert app.session_state.simulation.twin is None
    app.button(key="step").click().run()
    assert not app.exception
    original = app.session_state.simulation.twin
    app.multiselect(key="visible_signals").set_value(["load_ratio"]).run()
    app.run()
    assert not app.exception
    assert app.session_state.simulation.twin == original
    for view in ("Telemetría", "Alertas", "Mantenimiento", "Resumen"):
        app.radio(key="navigation").set_value(view).run()
        assert not app.exception
        assert app.session_state.simulation.twin == original
    app.button(key="start").click().run()
    assert app.session_state.simulation.running
    assert app.session_state.simulation.twin == original
    app.run()
    assert app.session_state.simulation.twin == original
    app.button(key="pause").click().run()
    assert not app.session_state.simulation.running
    assert app.session_state.simulation.twin == original


def test_degraded_engine_renders_alert_and_history_without_advancing():
    simulation = MineTwin(SimulationConfig(scenario=Scenario.ENGINE_DEGRADATION))
    simulation.advance(180)
    observation = simulation.twin
    app = _simulation_app()
    app.session_state.simulation = simulation
    app.run()
    assert not app.exception
    assert len(app.error) == 1
    assert "Alerta de motor" in app.error[0].value
    assert app.session_state.simulation.twin == observation
    app.radio(key="navigation").set_value("Alertas").run()
    assert not app.exception
    assert len(app.error) == 1
    assert len(app.dataframe) == 1
    assert app.session_state.simulation.twin == observation


def test_reset_applies_scenario_and_reproduces_first_observation():
    app = _simulation_app().run()
    app.button(key="step").click().run()
    first_observation = app.session_state.simulation.twin
    reset = next(
        button for button in app.button if button.label == "Reiniciar escenario"
    )
    reset.click().run()
    assert app.session_state.simulation.twin is None
    app.button(key="step").click().run()
    assert app.session_state.simulation.twin == first_observation
    app.selectbox(key="scenario").set_value(Scenario.ENGINE_DEGRADATION)
    next(
        button for button in app.button if button.label == "Reiniciar escenario"
    ).click()
    app.run()
    assert not app.exception
    assert app.session_state.simulation.config.scenario == Scenario.ENGINE_DEGRADATION
    assert app.session_state.simulation.twin is None


def test_manual_maintenance_controls_do_not_advance_time_and_recover_after_duration():
    simulation = MineTwin(
        SimulationConfig(scenario=Scenario.ENGINE_DEGRADATION),
        maintenance_config=MaintenanceConfig(replacement_seconds=60),
    )
    simulation.advance(180)
    initial_time = simulation.time
    initial_hours = simulation.twin.observation.reading("operating_hours").value
    app = _simulation_app(20)
    app.session_state.simulation = simulation
    app.run()
    app.radio(key="navigation").set_value("Mantenimiento").run()
    app.button(key="create_order").click().run()
    assert not app.exception
    assert app.session_state.simulation.current_order.status == OrderStatus.OPEN
    assert app.session_state.simulation.time == initial_time
    app.button(key="start_order").click().run()
    assert not app.exception
    assert app.session_state.simulation.asset_status == AssetStatus.MAINTENANCE
    assert app.session_state.simulation.time == initial_time
    assert app.button(key="cancel_order").disabled
    app.button(key="step").click().run()
    assert not app.exception
    assert app.session_state.simulation.orders[-1].status == OrderStatus.COMPLETED
    assert app.session_state.simulation.asset_status == AssetStatus.AVAILABLE
    assert (
        app.session_state.simulation.twin.observation.reading("operating_hours").value
        == initial_hours
    )
    assert not app.button(key="create_order").disabled
