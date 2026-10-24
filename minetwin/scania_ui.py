import csv
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from minetwin.assistant_ui import render_assistant
from minetwin.data.scania import (
    SCANIA_CLASS_DESCRIPTIONS,
    DatasetSplit,
    ScaniaReplaySession,
    ScaniaReplayStore,
    ScaniaReplayView,
)
from minetwin.explanations import describe
from minetwin.exports import Explanation
from minetwin.learning import FeatureConfig, WindowVectorizer
from minetwin.learning.inference import TorchRiskPrognosticator
from minetwin.learning.metrics import expected_decision_costs
from minetwin.learning.provenance import verify_training_provenance
from minetwin.paths import DATASET_ROOT, RESULTS_ROOT, result_path
from minetwin.presentation_ui import render_plot, render_table
from minetwin.publication import PublishedCondition, RiskAssessment
from minetwin.report_ui import render_article_report
from minetwin.research_ui import render_learning_results
from minetwin.scania_explanation import ScaniaExplanationSource
from minetwin.views.feria import render_feria

REPLAY_INDEX = result_path("02_reproduccion", "index.json")
PHASE_ONE_SUMMARY = result_path("01_datos", "summary.json")
LEARNING_METADATA = result_path("03_cache", "metadata.json")
EDA_PATHS = (result_path("04_seleccion_modelo", "eda.json"),)
FEDERATION_PATHS = (result_path("07_federacion", "summary.json"),)
MODEL_ROOTS = (result_path("06_aprendizaje", "seed_100"),)
DEMONSTRATION_CASES = result_path("11_mejora", "07_demostracion", "demo_cases.csv")
DEMONSTRATION_MANIFEST = result_path("11_mejora", "07_demostracion", "manifest.json")
VIEWS = (
    "Demo feria",
    "Dashboard",
    "Datos y EDA",
    "Telemetría",
    "Riesgo predictivo",
    "Resultados del artículo",
    "Aprendizaje",
    "Federación",
    "Asistente",
)
REGIMES = {
    "fedprox": "FedProx",
    "fedavg": "FedAvg",
    "centralized": "Centralizado",
    "local": "Local",
}
CLASS_LABELS = dict(enumerate(SCANIA_CLASS_DESCRIPTIONS))
CONDITION_LABELS = {
    PublishedCondition.NORMAL: "Riesgo estimado bajo",
    PublishedCondition.WATCH: "Seguimiento sugerido",
    PublishedCondition.ALERT: "Riesgo estimado alto",
}


class PublishedRiskUnavailableError(OSError):
    pass


@st.cache_resource
def _store(dataset_root: str, index: str) -> ScaniaReplayStore:
    return ScaniaReplayStore(Path(dataset_root), Path(index))


@st.cache_resource
def _predictor(model: str, scaler: str) -> TorchRiskPrognosticator:
    return TorchRiskPrognosticator.load(Path(model), Path(scaler))


@st.cache_resource
def _published_risk_source() -> ScaniaExplanationSource:
    return ScaniaExplanationSource()


def render_scania() -> None:
    if st.session_state.get("scania_navigation") == "Demo feria":
        render_feria()
        return
    if not REPLAY_INDEX.is_file():
        st.html('<div class="mt-eyebrow">DATOS REALES / SCANIA COMPONENT X</div>')
        st.title("MineTwin · Análisis histórico de Component X")
        _render_pending_index()
        return
    store = _store(str(DATASET_ROOT), str(REPLAY_INDEX))
    view_name, split, node_id, vehicle_id, regime = _sidebar(store)
    if view_name == "Demo feria":
        render_feria()
        return
    st.html('<div class="mt-eyebrow">DATOS REALES / SCANIA COMPONENT X</div>')
    st.title("MineTwin · Análisis histórico de Component X")
    st.caption(
        f"{view_name} · Split {split.value} · Partición {node_id} · "
        "Fuente SCANIA Component X"
    )
    if view_name == "Dashboard":
        _render_dashboard(store, split, regime)
    elif view_name == "Datos y EDA":
        _render_eda(store, split)
    elif view_name == "Telemetría":
        session = _session(store, split, vehicle_id)
        _render_replay_controls(session)
        _render_telemetry(session.view)
    elif view_name == "Riesgo predictivo":
        session = _session(store, split, vehicle_id)
        _render_replay_controls(session)
        _render_risk(session, regime)
    elif view_name == "Resultados del artículo":
        render_article_report(RESULTS_ROOT)
    elif view_name == "Aprendizaje":
        render_learning_results(RESULTS_ROOT)
    elif view_name == "Asistente":
        session = _session(store, split, vehicle_id)
        _render_replay_controls(session)
        if session.position < session.readout_count - 1 and st.button(
            "Ir al último readout", key="scania_latest_readout"
        ):
            session.advance(session.readout_count)
        render_assistant(session.view, regime)
    else:
        _render_federation(split)


def _sidebar(
    store: ScaniaReplayStore,
) -> tuple[str, DatasetSplit, str, str, str]:
    with st.sidebar:
        st.caption("NAVEGACIÓN SCANIA")
        view_name = st.radio(
            "Vista",
            VIEWS,
            label_visibility="collapsed",
            key="scania_navigation",
        )
        st.divider()
        st.caption("FUENTE REAL")
        split = st.selectbox(
            "Conjunto",
            tuple(DatasetSplit),
            index=1,
            format_func=lambda item: item.value.capitalize(),
            key="scania_split",
        )
        node_id = st.selectbox(
            "Partición experimental", store.nodes(split), key="scania_node"
        )
        vehicles = store.vehicles(split, node_id)
        vehicle_id = st.selectbox(
            "Vehículo",
            tuple(vehicle.vehicle_id for vehicle in vehicles),
            key="scania_vehicle",
        )
        regime = st.selectbox(
            "Modelo",
            tuple(REGIMES),
            format_func=REGIMES.get,
            key="scania_regime",
        )
        st.caption(
            "Lecturas históricas · Variables anónimas · "
            "Particiones derivadas de especificaciones"
        )
    return view_name, split, node_id, vehicle_id, regime


def _session(
    store: ScaniaReplayStore,
    split: DatasetSplit,
    vehicle_id: str,
) -> ScaniaReplaySession:
    metadata = _read_json(LEARNING_METADATA)
    window_size = metadata["feature_config"]["window_size"] if metadata else 12
    session = st.session_state.get("scania_replay")
    if (
        not isinstance(session, ScaniaReplaySession)
        or session.store is not store
        or session.window_size != window_size
    ):
        session = ScaniaReplaySession(
            store, split, vehicle_id=vehicle_id, window_size=window_size
        )
        st.session_state.scania_replay = session
    elif (session.split, session.vehicle_id) != (split, vehicle_id):
        session.select(split, vehicle_id)
    return session


def _render_dashboard(
    store: ScaniaReplayStore,
    split: DatasetSplit,
    regime: str,
) -> None:
    summary = _read_json(PHASE_ONE_SUMMARY)
    metadata = _read_json(LEARNING_METADATA)
    if summary is None:
        st.warning("No se encontró el resumen validado de SCANIA.")
        return
    split_summary = _record(summary["splits"], "split", split.value)
    vehicles = store.vehicles(split)
    missing_rate = split_summary["missing_values"] / (
        split_summary["readouts"] * split_summary["features"]
    )
    try:
        model_root = _model_root(regime, vehicles[0].node_id)
    except (OSError, ValueError):
        model_root = None
    columns = st.columns(5)
    columns[0].metric("Vehículos", f"{len(vehicles):,}")
    columns[1].metric("Readouts", f"{split_summary['readouts']:,}")
    columns[2].metric("Variables", split_summary["features"])
    columns[3].metric("Valores ausentes", f"{missing_rate:.2%}")
    columns[4].metric("Particiones", len(store.nodes(split)))
    _evidence(
        split,
        len(vehicles),
        "observado",
        "Conteos del manifiesto validado y de las particiones por vehículo.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Vehículos por partición")
        node_counts = {
            node: len(store.vehicles(split, node)) for node in store.nodes(split)
        }
        _bar_chart(node_counts, "Partición", "Vehículos")
        _evidence(
            split,
            len(vehicles),
            "derivado",
            "Cada vehículo permanece asignado a una única partición experimental.",
        )
    with right:
        st.subheader("Clases disponibles")
        learning_split = _learning_split(metadata, split)
        classes = learning_split.get("classes", {}) if learning_split else {}
        classes = {key: value for key, value in classes.items() if value}
        if classes:
            _bar_chart(
                {f"Clase {key}": value for key, value in classes.items()},
                "Clase",
                "Ejemplos causales",
            )
            _evidence(
                split,
                learning_split["examples"],
                "derivado",
                "Las clases proceden de las etiquetas oficiales y ventanas causales.",
            )
        else:
            st.info("El split de prueba no incluye etiquetas públicas.")
    st.subheader("Contexto activo")
    context = [
        {"Elemento": "Fuente", "Valor": "SCANIA Component X"},
        {"Elemento": "Split", "Valor": split.value},
        {"Elemento": "Régimen", "Valor": REGIMES[regime]},
        {
            "Elemento": "Artefacto del modelo",
            "Valor": model_root.name if model_root else "No disponible",
        },
        {"Elemento": "Naturaleza", "Valor": "Datos reales y resultados derivados"},
    ]
    render_table(
        context,
        hide_index=True,
        width="stretch",
        explanation=describe("context", "Contexto activo"),
        key="scania_ui_render_dashboard0",
    )
    _evidence(split, len(vehicles), "mixto", "No contiene variables simuladas.")


def _render_eda(store: ScaniaReplayStore, split: DatasetSplit) -> None:
    summary = _read_json(PHASE_ONE_SUMMARY)
    metadata = _read_json(LEARNING_METADATA)
    eda = _first_json(EDA_PATHS)
    if summary is None or metadata is None:
        st.warning("Faltan los artefactos de preparación del aprendizaje.")
        return
    learning_split = _learning_split(metadata, split)
    st.subheader("Distribución de clases")
    classes = learning_split.get("classes", {}) if learning_split else {}
    classes = {key: value for key, value in classes.items() if value}
    if classes:
        _bar_chart(
            {f"Clase {key}": value for key, value in classes.items()},
            "Clase",
            "Ejemplos causales",
        )
        _evidence(
            split,
            learning_split["examples"],
            "derivado",
            "La clase 0 domina; la exactitud simple no describe bien "
            "las clases escasas.",
        )
    else:
        st.info("No existen etiquetas públicas para este split.")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Vehículos por partición")
        nodes = {node: len(store.vehicles(split, node)) for node in store.nodes(split)}
        _bar_chart(nodes, "Partición", "Vehículos")
        _evidence(
            split,
            sum(nodes.values()),
            "derivado",
            "La diferencia de tamaños representa heterogeneidad entre particiones.",
        )
    with right:
        st.subheader("Ausencia por split")
        missing = {
            row["split"]: row["missing_values"] / (row["readouts"] * row["features"])
            for row in summary["splits"]
        }
        _bar_chart(missing, "Split", "Proporción ausente", percent=True)
        samples = " · ".join(
            f"{row['split']}: {row['readouts']:,}" for row in summary["splits"]
        )
        st.caption(
            f"Fuente: SCANIA Component X · Muestra de readouts: {samples} · "
            "Naturaleza: observado antes de imputación · Los valores ausentes "
            "no son ceros."
        )
    st.subheader("Significado de las clases")
    class_rows = [
        {"Clase": label, "Ventana hasta reparación": description}
        for label, description in CLASS_LABELS.items()
    ]
    render_table(
        class_rows,
        hide_index=True,
        width="stretch",
        explanation=describe("class_definitions", "Significado de las clases"),
        key="scania_ui_render_eda0",
    )
    _evidence(
        split,
        "No aplica",
        "definición",
        "Las clases representan ventanas temporales de Component X.",
    )
    st.subheader("Variables con mayor separación descriptiva")
    features = eda.get("top_class_separation_features", []) if eda else []
    if not features:
        st.info("El EDA avanzado se genera durante la selección de modelos.")
        return
    rows = [
        {
            "Variable derivada": item["feature"],
            "Separación η²": item["eta_squared"],
        }
        for item in features[:10]
    ]
    render_table(
        rows,
        hide_index=True,
        width="stretch",
        explanation=describe(
            "separation", "Variables con mayor separación descriptiva"
        ),
        key="scania_ui_render_eda1",
    )
    train = _learning_split(metadata, DatasetSplit.TRAIN)
    _evidence(
        DatasetSplit.TRAIN,
        train["examples"],
        "derivado",
        "η² mide asociación descriptiva entre variable y clase, no causalidad.",
    )


def _render_replay_controls(session: ScaniaReplaySession) -> None:
    view = session.view
    controls = st.columns(3)
    controls[0].button(
        "Anterior",
        on_click=session.retreat,
        disabled=view.position == 0,
        key="scania_previous",
        width="stretch",
    )
    controls[1].button(
        "Reiniciar",
        on_click=session.reset,
        disabled=view.position == 0,
        key="scania_reset",
        width="stretch",
    )
    controls[2].button(
        "Siguiente",
        on_click=session.advance,
        disabled=view.position == view.readout_count - 1,
        key="scania_next",
        type="primary",
        width="stretch",
    )


def _render_telemetry(view: ScaniaReplayView) -> None:
    missing = sum(value is None for value in view.current.values)
    columns = st.columns(5)
    columns[0].metric("Vehículo", view.vehicle_id)
    columns[1].metric("Partición", view.node_id)
    columns[2].metric("Readout", f"{view.position + 1}/{view.readout_count}")
    columns[3].metric("Paso temporal", f"{view.current.time_step:g}")
    columns[4].metric("Ausentes", f"{missing}/{len(view.current.values)}")
    _evidence(
        view.split,
        len(view.history),
        "observado",
        "La secuencia contiene solo el readout actual y sus antecedentes.",
    )
    selected = st.multiselect(
        "Variables visibles",
        view.current.feature_names,
        default=view.current.feature_names[:3],
        max_selections=6,
        key="scania_visible_features",
    )
    if selected:
        indices = {name: view.current.feature_names.index(name) for name in selected}
        figure = go.Figure()
        for name, index in indices.items():
            figure.add_trace(
                go.Scatter(
                    x=[item.time_step for item in view.history],
                    y=[item.values[index] for item in view.history],
                    mode="lines+markers",
                    name=name,
                    connectgaps=False,
                )
            )
        figure.update_layout(
            height=340,
            xaxis_title="Paso temporal",
            yaxis_title="Valor observado",
            legend_orientation="h",
        )
        _plot(figure, describe("telemetry", "Historia de las variables seleccionadas"))
        _evidence(
            view.split,
            len(view.history),
            "observado",
            "Los huecos corresponden a valores ausentes del archivo original.",
        )
    _render_window(view)
    rows = [
        {
            "Variable": name,
            "Valor observado": value,
            "Estado": "Ausente" if value is None else "Observado",
        }
        for name, value in zip(
            view.current.feature_names, view.current.values, strict=True
        )
    ]
    with st.expander("Variables del readout", expanded=False):
        render_table(
            rows,
            hide_index=True,
            width="stretch",
            explanation=describe("telemetry", "Variables del readout actual"),
            key="scania_ui_render_telemetry0",
        )
        _evidence(
            view.split,
            1,
            "observado",
            "Las 105 variables están anonimizadas por el proveedor.",
        )


def _render_window(view: ScaniaReplayView) -> None:
    window_cells = len(view.window.values) * len(view.window.feature_names)
    window_missing = sum(sum(row) for row in view.window.missing)
    st.subheader("Ventana causal del modelo")
    columns = st.columns(3)
    columns[0].metric("Observaciones", len(view.window.values))
    columns[1].metric("Inicio", f"{view.window.time_steps[0]:g}")
    columns[2].metric("Ausencia original", f"{window_missing / window_cells:.2%}")
    _evidence(
        view.split,
        len(view.window.values),
        "derivado",
        "La imputación LOCF conserva indicadores de ausencia y no utiliza el futuro.",
    )


def _render_risk(session: ScaniaReplaySession, regime: str) -> None:
    assessment_tab, demonstration_tab = st.tabs(
        ("Evaluación puntual", "Demostración de prioridad")
    )
    with assessment_tab:
        _render_live_risk(session, regime)
    with demonstration_tab:
        _render_priority_demonstration()


def _render_live_risk(session: ScaniaReplaySession, regime: str) -> None:
    view = session.view
    st.subheader("Estimación de riesgo de Component X")
    try:
        assessment, model_root, feature_count = _assessment(view, regime)
    except PublishedRiskUnavailableError as error:
        st.info(str(error))
        st.button(
            "Ir al último readout publicado",
            on_click=session.advance,
            args=(session.readout_count,),
            key="scania_published_risk",
            type="primary",
        )
        _render_window(view)
        return
    except (OSError, ValueError, RuntimeError) as error:
        st.warning(f"Estado no estimable: {error}")
        _render_window(view)
        return
    most_likely, highest_probability, decision_costs = _risk_decision(assessment)
    _risk_status(assessment)
    if view.split == DatasetSplit.TRAIN:
        st.caption(
            "Predicción exploratoria sobre entrenamiento; no constituye evidencia "
            "de generalización."
        )
    missing = sum(sum(row) for row in view.window.missing)
    cells = len(view.window.values) * len(view.window.feature_names)
    columns = st.columns(5)
    columns[0].metric(
        "Clase más probable",
        f"{most_likely} ({highest_probability:.2%})",
    )
    columns[1].metric(
        "Decisión publicada por costo",
        f"Clase {assessment.predicted_class}",
    )
    columns[2].metric(
        "Costo esperado publicado",
        f"{decision_costs[assessment.predicted_class]:.4f}",
    )
    columns[3].metric("Observaciones", len(view.window.values))
    columns[4].metric("Ausencia original", f"{missing / cells:.2%}")
    st.info(_risk_recommendation(assessment, most_likely, highest_probability))
    probabilities = {
        f"Clase {label}": probability
        for label, probability in enumerate(assessment.probabilities)
    }
    source = (
        f"SCANIA Component X · {view.split.value} · vehículo {view.vehicle_id} · "
        f"paso {view.current.time_step:g} · modelo {assessment.model_id}"
    )
    _bar_chart(
        probabilities,
        "Clase",
        "Probabilidad",
        percent=True,
        explanation=describe(
            "probabilities", "Probabilidades de la evaluación puntual", source=source
        ),
    )
    st.subheader("Costo esperado por decisión publicada")
    render_table(
        [
            {
                "Decisión": f"Clase {label}",
                "Costo esperado": float(cost),
                "Publicada": "Sí" if label == assessment.predicted_class else "",
            }
            for label, cost in enumerate(decision_costs)
        ],
        hide_index=True,
        width="stretch",
        explanation=describe(
            "decision_cost", "Costo esperado por decisión publicada", source=source
        ),
        key="scania_ui_render_live_risk0",
    )
    _evidence(
        view.split,
        len(view.window.values),
        "derivado",
        "La decisión minimiza el costo SCANIA; las barras son probabilidades "
        "del modelo.",
    )
    evidence = [
        {"Evidencia": "Vehículo", "Valor": str(view.vehicle_id)},
        {"Evidencia": "Partición", "Valor": str(view.node_id)},
        {"Evidencia": "Último paso observado", "Valor": f"{view.current.time_step:g}"},
        {"Evidencia": "Variables derivadas", "Valor": str(feature_count)},
        {"Evidencia": "Régimen", "Valor": str(REGIMES[regime])},
        {"Evidencia": "Modelo", "Valor": str(assessment.model_id)},
        {"Evidencia": "Artefacto", "Valor": str(model_root.name)},
        {"Evidencia": "Semilla", "Valor": str(_seed(model_root))},
        {
            "Evidencia": "Etiqueta retrospectiva",
            "Valor": str(_class_label(view.observed_class)),
        },
    ]
    render_table(
        evidence,
        hide_index=True,
        width="stretch",
        explanation=describe(
            "context", "Evidencia de la predicción activa", source=source
        ),
        key="scania_ui_render_live_risk1",
    )
    _evidence(
        view.split,
        1,
        "derivado",
        "No ejecuta reparaciones ni atribuye una pieza física a Component X.",
    )


def _render_priority_demonstration() -> None:
    rows = _read_csv(DEMONSTRATION_CASES)
    manifest = _read_json(DEMONSTRATION_MANIFEST)
    if not rows or manifest is None:
        st.info(
            "La demostración se habilita al generar ejemplos estratificados "
            "con predicciones fuera de fold."
        )
        return
    st.subheader("Cola de prioridad demostrativa")
    st.caption(
        "Ejemplos SCANIA reales de train con predicciones fuera de fold. "
        "No representan una evaluación final ni una orden de mantenimiento."
    )
    case = rows[
        st.selectbox(
            "Ejemplo",
            range(len(rows)),
            format_func=lambda index: _case_label(rows[index]),
            key="priority_demonstration_case",
        )
    ]
    selected = case["selected"] == "True"
    columns = st.columns(5)
    columns[0].metric("Vehículo", case["vehicle_id"])
    columns[1].metric("Ventana causal", case["window_observations"])
    columns[2].metric(
        "Clase más probable",
        f"{case['most_likely_class']} ({float(case['maximum_probability']):.2%})",
    )
    columns[3].metric("Prioridad", "Sí" if selected else "No")
    columns[4].metric(
        "Costo esperado publicado",
        f"{float(case['published_expected_cost']):.4f}",
    )
    if selected:
        st.warning(
            "Revisión humana prioritaria: el caso entra en la capacidad fijada "
            "por la política. No confirma una falla ni activa mantenimiento."
        )
    else:
        st.info(
            "No priorizado en la capacidad fijada. La evaluación humana sigue "
            "siendo necesaria para interpretar cualquier señal de riesgo."
        )
    probabilities = {
        f"Clase {label}": float(case[f"probability_{label}"]) for label in range(5)
    }
    source = (
        f"SCANIA Component X · train fuera de fold · vehículo {case['vehicle_id']} · "
        f"fold {case['fold']} · paso {case['time_step']} · {DEMONSTRATION_CASES.name}"
    )
    _bar_chart(
        probabilities,
        "Clase",
        "Probabilidad calibrada",
        percent=True,
        explanation=describe(
            "probabilities",
            "Probabilidades calibradas del ejemplo prioritario",
            source=source,
        ),
    )
    st.subheader("Costo esperado por decisión")
    render_table(
        [
            {
                "Decisión": f"Clase {label}",
                "Costo esperado": float(case[f"expected_cost_{label}"]),
                "Publicada": "Sí" if label == int(case["published_decision"]) else "",
            }
            for label in range(5)
        ],
        hide_index=True,
        width="stretch",
        explanation=describe(
            "decision_cost", "Costo esperado del ejemplo prioritario", source=source
        ),
        key="scania_ui_render_priority_demonstration0",
    )
    render_table(
        [
            {"Evidencia": "Estrato", "Valor": case["stratum"]},
            {"Evidencia": "Partición experimental", "Valor": case["node_id"]},
            {"Evidencia": "Paso temporal", "Valor": case["time_step"]},
            {"Evidencia": "Etiqueta retrospectiva", "Valor": case["label"]},
            {"Evidencia": "Fold fuera de fold", "Valor": case["fold"]},
            {"Evidencia": "Modelo", "Valor": manifest["priority"]["model"]},
            {"Evidencia": "Semilla", "Valor": manifest["priority"]["seed"]},
            {
                "Evidencia": "Capacidad de revisión",
                "Valor": manifest["priority"]["capacity"],
            },
        ],
        hide_index=True,
        width="stretch",
        explanation=describe(
            "context", "Evidencia del ejemplo fuera de fold", source=source
        ),
        key="scania_ui_render_priority_demonstration1",
    )


def _case_label(case: dict[str, str]) -> str:
    return (
        f"{case['stratum']} · vehículo {case['vehicle_id']} · "
        f"paso {float(case['time_step']):g}"
    )


def _risk_decision(assessment: RiskAssessment) -> tuple[int, float, tuple[float, ...]]:
    probabilities = assessment.probabilities
    most_likely = max(range(len(probabilities)), key=probabilities.__getitem__)
    costs = expected_decision_costs([probabilities])[0]
    return most_likely, probabilities[most_likely], tuple(float(cost) for cost in costs)


def _risk_recommendation(
    assessment: RiskAssessment,
    most_likely: int,
    highest_probability: float,
) -> str:
    return (
        "Recomendación: realizar una evaluación humana de Component X. "
        f"La clase más probable es {most_likely} ({highest_probability:.2%}), "
        f"mientras que la clase {assessment.predicted_class} se publica porque "
        "minimiza el costo esperado. La alerta no confirma una falla y la "
        "probabilidad máxima no es confianza en la decisión publicada."
    )


def _seed(model_root: Path) -> str:
    prefix = "seed_"
    return (
        model_root.name.removeprefix(prefix)
        if model_root.name.startswith(prefix)
        else "No disponible"
    )


def _assessment(
    view: ScaniaReplayView,
    regime: str,
) -> tuple[RiskAssessment, Path, int]:
    metadata = _read_json(LEARNING_METADATA)
    if metadata is None:
        raise OSError("no existe la configuración de variables del aprendizaje")
    model_root = _model_root(regime, view.node_id)
    if model_root is None:
        raise OSError(f"no existe un modelo {REGIMES[regime]} compatible")
    model_path = _model_path(model_root, regime, view.node_id)
    vectorizer = WindowVectorizer(FeatureConfig(**metadata["feature_config"]))
    features = vectorizer.transform(view.window)
    if len(features) != metadata["feature_count"]:
        raise ValueError("la ventana no coincide con el esquema entrenado")
    try:
        predictor = _predictor(str(model_path), str(model_root / "scaler.npz"))
    except OSError as error:
        return _published_assessment(view, regime, error), model_root, len(features)
    return predictor.assess(features[None, :])[0], model_root, len(features)


def _published_assessment(
    view: ScaniaReplayView, regime: str, native_error: OSError
) -> RiskAssessment:
    try:
        observation = _published_risk_source().observation(view, regime)
    except RuntimeError as error:
        raise PublishedRiskUnavailableError(
            "No se puede cargar el modelo nativo ni el estado publicado."
        ) from error
    if observation.state is None or observation.stale:
        raise PublishedRiskUnavailableError(
            "El estado publicado corresponde al último readout de validation "
            "con FedProx."
        ) from native_error
    return observation.state.risk


def _risk_status(assessment: RiskAssessment) -> None:
    message = CONDITION_LABELS[assessment.condition]
    with st.container(key=f"status_{assessment.condition.value}"):
        if assessment.condition == PublishedCondition.NORMAL:
            st.success(message)
        elif assessment.condition == PublishedCondition.WATCH:
            st.warning(message)
        else:
            st.error(message)


def _render_federation(active_split: DatasetSplit) -> None:
    data = _first_json(FEDERATION_PATHS)
    if data is None:
        st.warning("No existe una publicación federada para mostrar.")
        return
    split = DatasetSplit(data["split"])
    transfer = data["transfer"]
    provenance = data.get("provenance", {})
    if provenance.get("verified"):
        st.caption(
            f"Caché y modelos verificados · datos {provenance['data_version'][:12]} · "
            f"entrenamiento {provenance['training_id'][:12]}"
        )
    else:
        st.warning(
            "Esta publicación no contiene una huella verificable de caché y modelos."
        )
    st.caption(
        "Los nodos son particiones experimentales de una sola fuente pública. "
        "No se midió un despliegue entre contratistas."
    )
    if data.get("centralized_training_uses_raw_data"):
        st.warning(
            "El comparador centralizado reúne los datos de entrenamiento; "
            "la retención local no aplica a ese régimen."
        )
    columns = st.columns(5)
    columns[0].metric("Estados publicados", f"{data['coordinator_states']:,}")
    columns[1].metric("Variables crudas publicadas", data["coordinator_raw_features"])
    columns[2].metric("Registros crudos publicados", transfer["raw_records"])
    columns[3].metric("Mensajes", f"{transfer['messages']:,}")
    columns[4].metric("Mensajes medidos y estimados", _bytes(transfer["payload_bytes"]))
    _evidence(
        split,
        data["coordinator_states"],
        "partición experimental",
        f"Publicación federada con {REGIMES.get(data['regime'], data['regime'])}.",
    )
    if active_split != split:
        st.info(
            f"El artefacto federado disponible corresponde al split {split.value}; "
            f"el selector actual está en {active_split.value}."
        )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Retención por partición")
        rows = [
            {
                "Partición": node,
                "Registros asignados": values["private_records"],
                "Estados publicados": values["published_states"],
                "Cobertura de lecturas": values.get("mean_observation_quality"),
            }
            for node, values in data["nodes"].items()
        ]
        render_table(
            rows,
            hide_index=True,
            width="stretch",
            explanation=describe("federation", "Publicación por nodo"),
            key="scania_ui_render_federation0",
        )
        _evidence(
            split,
            sum(row["Registros asignados"] for row in rows),
            "partición experimental",
            "Los resúmenes de ventana se asignan a particiones lógicas; "
            "no se publican en los estados. La cobertura es uno menos la tasa "
            "media de valores ausentes; no mide precisión predictiva.",
        )
    with right:
        st.subheader("Transferencias registradas")
        transfer_rows = [
            {
                "Tipo": kind,
                "Mensajes": values["messages"],
                "Datos": _bytes(values["payload_bytes"]),
                "Cálculo": ("Serializado" if kind == "published_state" else "Estimado"),
            }
            for kind, values in transfer["by_type"].items()
            if values["messages"]
        ]
        render_table(
            transfer_rows,
            hide_index=True,
            width="stretch",
            explanation=describe("federation", "Transferencia por tipo de mensaje"),
            key="scania_ui_render_federation1",
        )
        _evidence(
            split,
            transfer["messages"],
            "derivado",
            "El ledger separa estados serializados de bytes estimados de "
            "parámetros y distribución del modelo.",
        )
    reduction = data.get("publication_reduction_ratio")
    if reduction is not None:
        st.metric(
            "Reducción al publicar estados frente a centralizar variables",
            f"{reduction:.1%}",
        )
    parameters = transfer["by_type"].get("model_parameters", {})
    rounds = parameters.get("messages", 0) // max(len(data["nodes"]), 1)
    st.caption(
        f"{rounds} rondas registradas · {parameters.get('messages', 0)} envíos "
        f"de parámetros estimados · {_bytes(parameters.get('payload_bytes', 0))} · "
        "coordinador de estados sin datos operativos crudos"
    )


def _render_pending_index() -> None:
    summary = _read_json(PHASE_ONE_SUMMARY)
    st.warning("El índice de reproducción SCANIA todavía no está preparado.")
    if summary is None:
        return
    splits = {item["split"]: item for item in summary["splits"]}
    columns = st.columns(3)
    columns[0].metric(
        "Vehículos", f"{sum(row['vehicles'] for row in splits.values()):,}"
    )
    columns[1].metric(
        "Readouts", f"{sum(row['readouts'] for row in splits.values()):,}"
    )
    columns[2].metric("Variables", summary["features"])
    st.caption(
        "El dataset está validado. Falta preparar el índice local para consultar "
        "vehículos sin cargar todos los CSV en memoria."
    )


def _bar_chart(
    values: dict,
    x_title: str,
    y_title: str,
    percent: bool = False,
    explanation: Explanation | None = None,
) -> None:
    figure = go.Figure(
        go.Bar(
            x=list(values),
            y=list(values.values()),
            marker_color="#d9971a",
            text=list(values.values()),
            texttemplate="%{y:.1%}" if percent else "%{y:,}",
            textposition="outside",
        )
    )
    figure.update_layout(
        height=310,
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis_tickformat=".1%" if percent else None,
        showlegend=False,
    )
    kind = {
        "Vehículos": "partitions",
        "Ejemplos causales": "classes",
        "Proporción ausente": "missing",
        "Probabilidad": "probabilities",
        "Probabilidad calibrada": "probabilities",
    }[y_title]
    _plot(figure, explanation or describe(kind, f"{y_title} por {x_title.lower()}"))


def _plot(figure: go.Figure, explanation: Explanation) -> None:
    figure.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#5f6b7a"},
        xaxis_gridcolor="#e1e6ec",
        yaxis_gridcolor="#e1e6ec",
    )
    render_plot(figure, explanation=explanation)


def _evidence(
    split: DatasetSplit,
    sample: int | str,
    nature: str,
    detail: str,
) -> None:
    sample_label = f"{sample:,}" if isinstance(sample, int) else sample
    st.caption(
        f"Fuente: SCANIA Component X · Split: {split.value} · "
        f"Muestra: {sample_label} · Naturaleza: {nature} · {detail}"
    )


def _model_root(regime: str, node_id: str) -> Path | None:
    verified = MODEL_ROOTS[0]
    if (verified / "metrics.json").is_file():
        _verify_model_root(verified)
        return verified if _model_path(verified, regime, node_id).is_file() else None
    for root in MODEL_ROOTS:
        if (root / "scaler.npz").is_file() and _model_path(
            root, regime, node_id
        ).is_file():
            return root
    return None


def _verify_model_root(root: Path) -> None:
    report = _read_json(root / "metrics.json")
    if report is None:
        raise OSError("no existe el reporte de procedencia del modelo")
    verify_training_provenance(LEARNING_METADATA.parent, root, report)


def _model_path(root: Path, regime: str, node_id: str) -> Path:
    name = f"local_{node_id}.pt" if regime == "local" else f"{regime}.pt"
    return root / name


def _learning_split(data: dict | None, split: DatasetSplit) -> dict | None:
    if data is None:
        return None
    return _record(data["splits"], "split", split.value)


def _record(rows: list[dict], key: str, value: str) -> dict:
    return next(row for row in rows if row[key] == value)


def _first_json(paths: tuple[Path, ...]) -> dict | None:
    return next((data for path in paths if (data := _read_json(path))), None)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _class_label(value: int | None) -> str:
    return "No disponible" if value is None else f"{value} · {CLASS_LABELS[value]}"


def _bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if amount < 1024 or unit == "GB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise RuntimeError("Unidad de almacenamiento no disponible.")
