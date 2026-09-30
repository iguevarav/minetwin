import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from minetwin.data.scania import (
    DatasetSplit,
    ScaniaReplaySession,
    ScaniaReplayStore,
    ScaniaReplayView,
)
from minetwin.learning import FeatureConfig, WindowVectorizer
from minetwin.learning.inference import TorchRiskPrognosticator
from minetwin.publication import PublishedCondition, RiskAssessment
from minetwin.research_ui import render_learning_results

PROJECT_ROOT = Path(__file__).parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
DATASET_ROOT = Path(__file__).parent / "data" / "scania"
REPLAY_INDEX = RESULTS_ROOT / "phase1_replay" / "index.json"
PHASE_ONE_SUMMARY = RESULTS_ROOT / "phase1_scania" / "summary.json"
LEARNING_METADATA = RESULTS_ROOT / "phase2_cache" / "metadata.json"
EDA_PATHS = (
    RESULTS_ROOT / "final_selection" / "eda.json",
    RESULTS_ROOT / "model_selection" / "eda.json",
)
FEDERATION_PATHS = (
    RESULTS_ROOT / "final_federation" / "summary.json",
    RESULTS_ROOT / "final_federation_preselection" / "summary.json",
    RESULTS_ROOT / "phase3_federation" / "summary.json",
)
MODEL_ROOTS = (
    RESULTS_ROOT / "final_learning" / "seed_100",
    RESULTS_ROOT / "phase2_models",
)
VIEWS = (
    "Dashboard",
    "Datos y EDA",
    "Telemetría",
    "Riesgo predictivo",
    "Aprendizaje",
    "Federación",
)
REGIMES = {
    "fedprox": "FedProx",
    "fedavg": "FedAvg",
    "centralized": "Centralizado",
    "local": "Local",
}
CLASS_LABELS = {
    0: "Más de 48 pasos o sin reparación",
    1: "Entre 24 y 48 pasos",
    2: "Entre 12 y 24 pasos",
    3: "Entre 6 y 12 pasos",
    4: "Hasta 6 pasos",
}
CONDITION_LABELS = {
    PublishedCondition.NORMAL: "Operación normal",
    PublishedCondition.WATCH: "Seguimiento requerido",
    PublishedCondition.ALERT: "Condición crítica",
}


@st.cache_resource
def _store(dataset_root: str, index: str) -> ScaniaReplayStore:
    return ScaniaReplayStore(Path(dataset_root), Path(index))


@st.cache_resource
def _predictor(model: str, scaler: str) -> TorchRiskPrognosticator:
    return TorchRiskPrognosticator.load(Path(model), Path(scaler))


def render_scania() -> None:
    st.html('<div class="mt-eyebrow">DATOS REALES / SCANIA COMPONENT X</div>')
    st.title("MineTwin · Reproducción de datos reales")
    if not REPLAY_INDEX.is_file():
        _render_pending_index()
        return
    store = _store(str(DATASET_ROOT), str(REPLAY_INDEX))
    view_name, split, node_id, vehicle_id, regime = _sidebar(store)
    st.caption(
        f"{view_name} · Split {split.value} · Nodo {node_id} · "
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
        _render_risk(session.view, regime)
    elif view_name == "Aprendizaje":
        render_learning_results(RESULTS_ROOT)
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
        node_id = st.selectbox("Nodo", store.nodes(split), key="scania_node")
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
        st.caption("Datos observados · Variables físicas anónimas")
    return view_name, split, node_id, vehicle_id, regime


def _session(
    store: ScaniaReplayStore,
    split: DatasetSplit,
    vehicle_id: str,
) -> ScaniaReplaySession:
    session = st.session_state.get("scania_replay")
    if not isinstance(session, ScaniaReplaySession) or session.store is not store:
        session = ScaniaReplaySession(store, split, vehicle_id=vehicle_id)
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
    model_root = _model_root(regime, vehicles[0].node_id)
    columns = st.columns(5)
    columns[0].metric("Vehículos", f"{len(vehicles):,}")
    columns[1].metric("Readouts", f"{split_summary['readouts']:,}")
    columns[2].metric("Variables", split_summary["features"])
    columns[3].metric("Valores ausentes", f"{missing_rate:.2%}")
    columns[4].metric("Nodos", len(store.nodes(split)))
    _evidence(
        split,
        len(vehicles),
        "observado",
        "Conteos del manifiesto validado y de las particiones por vehículo.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Vehículos por nodo")
        node_counts = {
            node: len(store.vehicles(split, node)) for node in store.nodes(split)
        }
        _bar_chart(node_counts, "Nodo", "Vehículos")
        _evidence(
            split,
            len(vehicles),
            "derivado",
            "Cada vehículo permanece asignado a un único nodo.",
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
    st.dataframe(context, hide_index=True, width="stretch")
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
        st.subheader("Vehículos por nodo")
        nodes = {
            node: len(store.vehicles(split, node)) for node in store.nodes(split)
        }
        _bar_chart(nodes, "Nodo", "Vehículos")
        _evidence(
            split,
            sum(nodes.values()),
            "derivado",
            "La diferencia de tamaños representa heterogeneidad entre nodos.",
        )
    with right:
        st.subheader("Ausencia por split")
        missing = {
            row["split"]: row["missing_values"]
            / (row["readouts"] * row["features"])
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
    st.dataframe(class_rows, hide_index=True, width="stretch")
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
    st.dataframe(rows, hide_index=True, width="stretch")
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
    columns[1].metric("Nodo", view.node_id)
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
        _plot(figure)
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
        st.dataframe(rows, hide_index=True, width="stretch")
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


def _render_risk(view: ScaniaReplayView, regime: str) -> None:
    st.subheader("Estimación de riesgo de Component X")
    try:
        assessment, model_root, feature_count = _assessment(view, regime)
    except (OSError, ValueError, RuntimeError) as error:
        st.warning(f"Estado no estimable: {error}")
        _render_window(view)
        return
    _risk_status(assessment)
    if view.split == DatasetSplit.TRAIN:
        st.caption(
            "Predicción exploratoria sobre entrenamiento; no constituye evidencia "
            "de generalización."
        )
    missing = sum(sum(row) for row in view.window.missing)
    cells = len(view.window.values) * len(view.window.feature_names)
    columns = st.columns(5)
    columns[0].metric("Clase predicha", assessment.predicted_class)
    columns[1].metric("Estado", CONDITION_LABELS[assessment.condition])
    columns[2].metric("Confianza máxima", f"{max(assessment.probabilities):.2%}")
    columns[3].metric("Observaciones", len(view.window.values))
    columns[4].metric("Ausencia original", f"{missing / cells:.2%}")
    st.info(assessment.recommendation)
    probabilities = {
        f"Clase {label}": probability
        for label, probability in enumerate(assessment.probabilities)
    }
    _bar_chart(probabilities, "Clase", "Probabilidad", percent=True)
    _evidence(
        view.split,
        len(view.window.values),
        "derivado",
        "La decisión minimiza el costo SCANIA; las barras son probabilidades "
        "del modelo.",
    )
    evidence = [
        {"Evidencia": "Vehículo", "Valor": view.vehicle_id},
        {"Evidencia": "Nodo", "Valor": view.node_id},
        {"Evidencia": "Último paso observado", "Valor": view.current.time_step},
        {"Evidencia": "Variables derivadas", "Valor": feature_count},
        {"Evidencia": "Régimen", "Valor": REGIMES[regime]},
        {"Evidencia": "Modelo", "Valor": assessment.model_id},
        {"Evidencia": "Artefacto", "Valor": model_root.name},
        {
            "Evidencia": "Etiqueta de referencia",
            "Valor": _class_label(view.observed_class),
        },
    ]
    st.dataframe(evidence, hide_index=True, width="stretch")
    _evidence(
        view.split,
        1,
        "derivado",
        "No ejecuta reparaciones ni atribuye una pieza física a Component X.",
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
    predictor = _predictor(str(model_path), str(model_root / "scaler.npz"))
    vectorizer = WindowVectorizer(FeatureConfig(**metadata["feature_config"]))
    features = vectorizer.transform(view.window)
    if len(features) != metadata["feature_count"]:
        raise ValueError("la ventana no coincide con el esquema entrenado")
    return predictor.assess(features[None, :])[0], model_root, len(features)


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
    columns = st.columns(5)
    columns[0].metric("Estados publicados", f"{data['coordinator_states']:,}")
    columns[1].metric("Variables crudas centrales", data["coordinator_raw_features"])
    columns[2].metric("Registros crudos recibidos", transfer["raw_records"])
    columns[3].metric("Mensajes", f"{transfer['messages']:,}")
    columns[4].metric("Transferencia", _bytes(transfer["payload_bytes"]))
    _evidence(
        split,
        data["coordinator_states"],
        "derivado",
        f"Publicación federada con {REGIMES.get(data['regime'], data['regime'])}.",
    )
    if active_split != split:
        st.info(
            f"El artefacto federado disponible corresponde al split {split.value}; "
            f"el selector actual está en {active_split.value}."
        )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Retención por nodo")
        rows = [
            {
                "Nodo": node,
                "Registros privados": values["private_records"],
                "Estados publicados": values["published_states"],
            }
            for node, values in data["nodes"].items()
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        _evidence(
            split,
            sum(row["Registros privados"] for row in rows),
            "derivado",
            "Las ventanas con 105 variables permanecen en su nodo.",
        )
    with right:
        st.subheader("Transferencias registradas")
        transfer_rows = [
            {
                "Tipo": kind,
                "Mensajes": values["messages"],
                "Datos": _bytes(values["payload_bytes"]),
            }
            for kind, values in transfer["by_type"].items()
            if values["messages"]
        ]
        st.dataframe(transfer_rows, hide_index=True, width="stretch")
        _evidence(
            split,
            transfer["messages"],
            "derivado",
            "El ledger separa estados, parámetros y distribuciones del modelo.",
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
        f"de parámetros · {_bytes(parameters.get('payload_bytes', 0))} · "
        "Coordinador sin datos operativos crudos"
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
    _plot(figure)


def _plot(figure: go.Figure) -> None:
    figure.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#5f6b7a"},
        xaxis_gridcolor="#e1e6ec",
        yaxis_gridcolor="#e1e6ec",
    )
    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
    )


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
    for root in MODEL_ROOTS:
        if (root / "scaler.npz").is_file() and _model_path(
            root, regime, node_id
        ).is_file():
            return root
    return None


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


def _class_label(value: int | None) -> str:
    return "No disponible" if value is None else f"{value} · {CLASS_LABELS[value]}"


def _bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if amount < 1024 or unit == "GB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise RuntimeError("Unidad de almacenamiento no disponible.")
