import csv
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from minetwin.explanations import describe
from minetwin.exports import Explanation
from minetwin.paths import RESULTS_ROOT
from minetwin.presentation_ui import render_plot, render_table

REGIME_LABELS = {
    "local": "Local",
    "centralized": "Centralizado",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
}


def render_learning_results(root: Path = RESULTS_ROOT) -> None:
    st.caption("VALIDACIÓN CON DATOS REALES")
    selection = _selection_path(root)
    final = root / "06_aprendizaje"
    diagnostics = root / "08_diagnostico"
    interpretability = root / "09_interpretabilidad"
    tabs = st.tabs(
        (
            "EDA",
            "Entrenamiento",
            "Diagnóstico predictivo",
            "Selección e hiperparámetros",
            "Validación cruzada",
            "Interpretabilidad",
            "Estadística",
            "Soberanía",
        )
    )
    with tabs[0]:
        _render_dataset(root / "01_datos" / "summary.json")
        _render_eda(selection / "eda.json")
    with tabs[1]:
        if (final / "metrics.csv").is_file():
            _render_multiseed_learning(final)
    with tabs[2]:
        _render_diagnostics(diagnostics / "diagnostics.json")
    with tabs[3]:
        _render_model_selection(selection)
    with tabs[4]:
        _render_cross_validation(selection)
    with tabs[5]:
        _render_interpretability(interpretability)
    with tabs[6]:
        _render_statistics(final)
    with tabs[7]:
        _render_sovereignty(root / "07_federacion" / "summary.json")


def _verified_learning_path(root: Path) -> Path | None:
    path = root / "06_aprendizaje"
    manifest = _json(path / "manifest.json")
    if manifest is not None and (path / "seed_100" / "metrics.json").is_file():
        return path
    return None


def _render_dataset(path: Path) -> None:
    data = _json(path)
    if data is None:
        st.info("Ejecuta la preparación de SCANIA para mostrar el conjunto de datos.")
        return
    splits = {item["split"]: item for item in data["splits"]}
    columns = st.columns(4)
    columns[0].metric("Variables", data["features"])
    columns[1].metric("Vehículos de entrenamiento", f"{splits['train']['vehicles']:,}")
    columns[2].metric(
        "Vehículos de validación", f"{splits['validation']['vehicles']:,}"
    )
    columns[3].metric("Nodos", len(data["nodes"]))
    _interpretation(
        measure="Tamaño y separación de los conjuntos reales usados por el estudio.",
        direction="No tiene dirección óptima; verifica cobertura y ausencia de mezcla.",
        result=(
            f"{splits['train']['vehicles']:,} vehículos de entrenamiento y "
            f"{splits['validation']['vehicles']:,} de validación, con "
            f"{data['features']} variables."
        ),
        conclusion="Los conjuntos oficiales contienen vehículos distintos.",
        limitation="Las variables son anónimas y solo representan Component X.",
    )


def _render_eda(path: Path) -> None:
    data = _json(path)
    st.subheader("Análisis exploratorio")
    if data is None:
        st.info("Ejecuta la selección de modelos para generar el EDA reproducible.")
        return
    train = next(item for item in data["splits"] if item["split"] == "train")
    classes = train["classes"]
    figure = go.Figure(
        go.Bar(
            x=[f"Clase {label}" for label in classes],
            y=list(classes.values()),
            marker_color="#f2b544",
            text=list(classes.values()),
            textposition="outside",
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Ejemplos",
        showlegend=False,
    )
    _plot(figure, describe("classes", "Frecuencia de clases de entrenamiento"))
    dominant_label, dominant_count = max(classes.items(), key=lambda item: item[1])
    _interpretation(
        measure="Cantidad de ejemplos causales disponibles para cada clase de riesgo.",
        direction="Una distribución más equilibrada reduce el sesgo hacia la mayoría.",
        result=(
            f"La clase {dominant_label} reúne {dominant_count:,} de "
            f"{train['examples']:,} ejemplos "
            f"({dominant_count / train['examples']:.1%})."
        ),
        conclusion=(
            "El desbalance exige costo, F1 macro y exactitud balanceada; la exactitud "
            "simple puede ocultar errores críticos."
        ),
        limitation="La distribución describe la muestra y no el riesgo poblacional.",
    )
    distribution = [
        {
            "Nodo": node,
            "Ejemplos": count,
            "Participación": count / train["examples"],
        }
        for node, count in train["nodes"].items()
    ]
    render_table(
        distribution,
        hide_index=True,
        width="stretch",
        explanation=describe("partitions", "Ejemplos por nodo"),
        key="research_ui_render_eda0",
    )
    largest = max(distribution, key=lambda row: row["Ejemplos"])
    smallest = min(distribution, key=lambda row: row["Ejemplos"])
    _interpretation(
        measure="Cantidad y participación de ejemplos privados por nodo federado.",
        direction=(
            "Tamaños similares reducen la influencia desproporcionada de un nodo."
        ),
        result=(
            f"{largest['Nodo']} contiene {largest['Ejemplos']:,} ejemplos y "
            f"{smallest['Nodo']} {smallest['Ejemplos']:,}."
        ),
        conclusion="Existe heterogeneidad de volumen entre los nodos experimentales.",
        limitation=(
            "Los nodos son particiones experimentales, no contratistas observados."
        ),
    )
    missing = data.get("raw_missing_rate", {})
    if missing:
        render_table(
            [
                {"Conjunto": split, "Proporción ausente": value}
                for split, value in missing.items()
            ],
            hide_index=True,
            width="stretch",
            explanation=describe("missing", "Ausencia por conjunto"),
            key="research_ui_render_eda1",
        )
        worst_split, worst_rate = max(missing.items(), key=lambda item: item[1])
        _interpretation(
            measure="Proporción de celdas ausentes antes de la imputación causal.",
            direction="Una proporción menor conserva más evidencia observada.",
            result=f"El máximo corresponde a {worst_split}: {worst_rate:.2%}.",
            conclusion=(
                "La ausencia se conserva como indicador y no se trata como cero."
            ),
            limitation=(
                "La tasa global puede ocultar concentración en variables concretas."
            ),
        )
    separation = [
        {
            "Variable derivada": item["feature"],
            "Separación entre clases η²": item["eta_squared"],
        }
        for item in data["top_class_separation_features"]
    ]
    render_table(
        separation,
        hide_index=True,
        width="stretch",
        explanation=describe("separation", "Separación η² por variable"),
        key="research_ui_render_eda2",
    )
    strongest = separation[0]
    _interpretation(
        measure="Proporción de variación asociada descriptivamente con la clase.",
        direction="Un η² mayor indica más separación entre clases.",
        result=(
            f"{strongest['Variable derivada']} obtuvo el mayor η²: "
            f"{strongest['Separación entre clases η²']:.4f}."
        ),
        conclusion="La variable encabeza la separación descriptiva del conjunto.",
        limitation=(
            "η² no demuestra causalidad ni importancia para el modelo entrenado."
        ),
    )


def _render_preliminary_learning(path: Path) -> None:
    report = _json(path)
    st.subheader("Rendimiento de los modelos")
    if report is None:
        st.info("Ejecuta el entrenamiento para mostrar resultados.")
        return
    rows = [
        {
            "Régimen": REGIME_LABELS.get(regime, regime),
            "Costo": values["global"]["total_cost"],
            "F1 macro": values["global"]["macro_f1"],
            "Exactitud balanceada": values["global"]["balanced_accuracy"],
            "PR AUC macro": values["global"]["macro_pr_auc"],
        }
        for regime, values in report["results"].items()
    ]
    st.caption(
        "Resultado preliminar de una semilla; no representa inferencia estadística."
    )
    _learning_chart(rows)
    best_balance = max(rows, key=lambda row: row["Exactitud balanceada"])
    _interpretation(
        measure="Exactitud balanceada, promedio del recall de las cinco clases.",
        direction="Un valor más alto representa mejor equilibrio entre clases.",
        result=(
            f"{best_balance['Régimen']} obtuvo la mayor exactitud balanceada: "
            f"{best_balance['Exactitud balanceada']:.4f}."
        ),
        conclusion="Ese régimen mostró el mejor equilibrio en esta ejecución.",
        limitation="Una sola semilla no permite inferencia estadística.",
    )
    render_table(
        rows,
        hide_index=True,
        width="stretch",
        explanation=describe("performance", "Rendimiento de una semilla"),
        key="research_ui_render_preliminary_learning0",
    )
    best_cost = min(rows, key=lambda row: row["Costo"])
    _interpretation(
        measure="Costo total según la matriz oficial de errores SCANIA.",
        direction="Un costo menor penaliza menos los errores críticos.",
        result=(
            f"{best_cost['Régimen']} obtuvo el menor costo: {best_cost['Costo']:.2f}."
        ),
        conclusion="Es el mejor resultado operativo observado en esta semilla.",
        limitation="Debe confirmarse con semillas pareadas e incertidumbre.",
    )
    _render_training_history(report)


def _render_multiseed_learning(path: Path) -> None:
    metrics = _csv(path / "metrics.csv")
    rows = []
    for regime in sorted({row["regime"] for row in metrics}):
        selected = [row for row in metrics if row["regime"] == regime]
        rows.append(
            {
                "Régimen": REGIME_LABELS.get(regime, regime),
                "Costo": sum(float(row["mean_cost"]) for row in selected)
                / len(selected),
                "Exactitud balanceada": sum(
                    float(row["balanced_accuracy"]) for row in selected
                )
                / len(selected),
                "PR AUC macro": sum(float(row["macro_pr_auc"]) for row in selected)
                / len(selected),
                "Exactitud": sum(float(row["accuracy"]) for row in selected)
                / len(selected),
            }
        )
    st.subheader("Rendimiento de los modelos")
    st.caption("10 semillas de evaluación pareadas.")
    _learning_chart(rows)
    best_balance = max(rows, key=lambda row: row["Exactitud balanceada"])
    _interpretation(
        measure="Exactitud balanceada media sobre las mismas semillas y validación oficial.",
        direction="Un valor mayor indica mejor equilibrio entre las cinco clases.",
        result=(
            f"{best_balance['Régimen']} alcanzó "
            f"{best_balance['Exactitud balanceada']:.4f}."
        ),
        conclusion="Fue el mayor equilibrio medio observado entre los regímenes.",
        limitation="La superioridad requiere intervalo y valor p ajustado.",
    )
    render_table(
        rows,
        hide_index=True,
        width="stretch",
        explanation=describe("performance", "Rendimiento medio de diez semillas"),
        key="research_ui_render_multiseed_learning0",
    )
    best = min(rows, key=lambda row: row["Costo"])
    st.caption(
        f"Menor costo medio observado: {best['Régimen']} ({best['Costo']:.2f}). "
        "Revisar baselines, errores por clase y pruebas pareadas antes de "
        "concluir utilidad."
    )
    _interpretation(
        measure="Costo, exactitud balanceada, PR AUC y exactitud medios.",
        direction="Menor costo y mayores métricas restantes son favorables.",
        result=f"{best['Régimen']} presentó el menor costo medio: {best['Costo']:.2f}.",
        conclusion="Es el mejor costo observado sobre la validación oficial.",
        limitation=(
            "Las métricas son bajas y deben leerse junto con la inferencia estadística."
        ),
    )
    artifacts = sorted(path.glob("seed_*/metrics.json"))
    if artifacts:
        _render_training_history(_json(artifacts[0]))


def _learning_chart(rows: list[dict]) -> None:
    figure = go.Figure(
        go.Bar(
            x=[row["Régimen"] for row in rows],
            y=[row["Exactitud balanceada"] for row in rows],
            marker_color="#f2b544",
            hovertemplate="%{x}<br>Exactitud balanceada %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Exactitud balanceada",
        showlegend=False,
    )
    _plot(figure, describe("performance", "Exactitud balanceada por régimen"))


def _render_diagnostics(path: Path) -> None:
    report = _json(path)
    st.subheader("Diagnóstico predictivo")
    if report is None:
        st.info(
            "Genera el diagnóstico sobre una semilla entrenada para ver "
            "baselines, calibración y errores."
        )
        return
    st.caption(
        f"Validación oficial SCANIA · semilla {report['training_config']['seed']}. "
        "Diagnóstico descriptivo. "
        "La selección de hiperparámetros se hizo solo con entrenamiento."
    )
    train = report["training_class_counts"]
    validation = report["validation_class_counts"]
    render_table(
        [
            {
                "Clase": index,
                "Entrenamiento": train[index],
                "Proporción entrenamiento": train[index] / sum(train),
                "Validación": validation[index],
                "Proporción validación": validation[index] / sum(validation),
            }
            for index in range(5)
        ],
        hide_index=True,
        width="stretch",
        explanation=describe("classes", "Clases en entrenamiento y validación"),
        key="research_ui_render_diagnostics0",
    )
    _interpretation(
        measure=(
            "Frecuencia relativa de cada clase antes y después de la "
            "separación oficial."
        ),
        direction="Las proporciones similares facilitan transportar el modelo.",
        result=(
            f"La clase 0 pasa de {train[0] / sum(train):.1%} en entrenamiento "
            f"a {validation[0] / sum(validation):.1%} en validación."
        ),
        conclusion=(
            "Existe un cambio de distribución que afecta la interpretación "
            "de las métricas."
        ),
        limitation="La diferencia descriptiva no identifica por sí sola su causa.",
    )
    baseline_labels = {
        "always_0": "Siempre clase 0",
        "always_4": "Siempre clase 4",
        "train_prior": "Prior de entrenamiento",
    }
    baselines = report["baselines"]
    render_table(
        [
            {
                "Baseline": baseline_labels[key],
                "Costo medio": item["global"]["mean_cost"],
                "F1 macro": item["global"]["macro_f1"],
                "Exactitud balanceada": item["global"]["balanced_accuracy"],
                "PR AUC macro": item["global"]["macro_pr_auc"],
            }
            for key, item in baselines.items()
        ],
        hide_index=True,
        width="stretch",
        explanation=describe("performance", "Referencias sin red neuronal"),
        key="research_ui_render_diagnostics1",
    )
    best_baseline = min(
        baselines, key=lambda key: baselines[key]["global"]["mean_cost"]
    )
    _interpretation(
        measure="Costo y clasificación de tres reglas sin entrenamiento de red.",
        direction=(
            "Costo menor y F1 macro mayor indican mejora frente a reglas triviales."
        ),
        result=(
            f"{baseline_labels[best_baseline]} logra el menor costo basal: "
            f"{baselines[best_baseline]['global']['mean_cost']:.3f} por ejemplo."
        ),
        conclusion=(
            "El modelo debe compararse con esta referencia antes de reclamar utilidad."
        ),
        limitation=(
            "Los baselines se definieron sin elegir una regla sobre la validación."
        ),
    )
    regime = st.selectbox(
        "Régimen para examinar",
        tuple(report["results"]),
        format_func=lambda value: REGIME_LABELS.get(value, value),
        key="diagnostic_regime",
    )
    selected = report["results"][regime]
    global_metrics = selected["global"]
    render_table(
        [
            {
                "Costo medio": global_metrics["mean_cost"],
                "F1 macro": global_metrics["macro_f1"],
                "Exactitud balanceada": global_metrics["balanced_accuracy"],
                "PR AUC macro": global_metrics["macro_pr_auc"],
                "Brier multiclase": global_metrics["brier_score"],
                "ECE top-1": global_metrics["argmax_ece"],
            }
        ],
        hide_index=True,
        width="stretch",
        explanation=describe("calibration", "Métricas globales y calibración"),
        key="research_ui_render_diagnostics2",
    )
    argmax = report["argmax_results"][regime]
    render_table(
        [
            {
                "Decisión": name,
                "Costo medio": metrics["mean_cost"],
                "F1 macro": metrics["macro_f1"],
                "Exactitud balanceada": metrics["balanced_accuracy"],
            }
            for name, metrics in (
                ("Matriz de costo", global_metrics),
                ("Mayor probabilidad", argmax),
            )
        ],
        hide_index=True,
        width="stretch",
        explanation=describe("performance", "Comparación de reglas de decisión"),
        key="research_ui_render_diagnostics3",
    )
    _interpretation(
        measure="Dos reglas de decisión aplicadas a las mismas probabilidades.",
        direction="Menor costo y mayor equilibrio entre clases son favorables.",
        result=(
            f"Costo medio: matriz {global_metrics['mean_cost']:.3f}; "
            f"máxima probabilidad {argmax['mean_cost']:.3f}."
        ),
        conclusion=(
            "La matriz penaliza más los errores sobre clases próximas a reparación."
        ),
        limitation=(
            "Esta comparación es descriptiva; no selecciona una regla con "
            "validación oficial."
        ),
    )
    baseline_cost = baselines[best_baseline]["global"]["mean_cost"]
    baseline_f1 = max(item["global"]["macro_f1"] for item in baselines.values())
    if (
        global_metrics["mean_cost"] >= baseline_cost
        or global_metrics["macro_f1"] <= baseline_f1
    ):
        st.warning(
            "Este resultado no supera simultáneamente el costo del mejor baseline "
            "y el F1 macro basal. No demuestra calidad práctica para mantenimiento."
        )
    else:
        st.info(
            "Supera las referencias triviales en esta semilla; falta validar "
            "estabilidad estadística y utilidad en una operación minera."
        )
    _interpretation(
        measure=(
            "Costo de errores, calidad por clase, precisión-recall y calibración "
            "de las probabilidades sin recalibración posterior."
        ),
        direction="Menor costo, Brier y ECE; mayor F1, exactitud balanceada y PR AUC.",
        result=(
            f"{REGIME_LABELS[regime]}: costo {global_metrics['mean_cost']:.3f}, "
            f"F1 {global_metrics['macro_f1']:.3f}, "
            f"Brier {global_metrics['brier_score']:.3f}, "
            f"ECE {global_metrics['argmax_ece']:.3f}."
        ),
        conclusion=(
            "Una probabilidad confiable requiere Brier y ECE bajos además de "
            "buena clasificación."
        ),
        limitation=(
            "ECE agrupa la confianza de la clase más probable en diez intervalos; "
            "no mide la calibración de la decisión por costo. Esta semilla no "
            "permite inferencia."
        ),
    )
    render_table(
        [
            {
                "Confianza": f"{item['lower']:.0%}–{item['upper']:.0%}",
                "Ejemplos": item["examples"],
                "Confianza media": item["mean_confidence"],
                "Exactitud top-1": item["accuracy"],
            }
            for item in global_metrics["calibration_bins"]
            if item["examples"]
        ],
        hide_index=True,
        width="stretch",
        explanation=describe("calibration", "Intervalos de calibración"),
        key="research_ui_render_diagnostics4",
    )
    _interpretation(
        measure="Exactitud observada frente a confianza media por intervalo.",
        direction="Ambas cifras deben ser cercanas en cada intervalo.",
        result=f"ECE top-1: {global_metrics['argmax_ece']:.3f}.",
        conclusion=(
            "Los intervalos muestran dónde se concentra el error de calibración."
        ),
        limitation=(
            "La exactitud top-1 difiere de la clase elegida por la matriz de costo."
        ),
    )
    _render_diagnostic_confusion(global_metrics)
    render_table(
        [
            {
                "Nodo experimental": node,
                "Ejemplos": values["examples"],
                "Costo medio": values["mean_cost"],
                "F1 macro": values["macro_f1"],
                "Exactitud balanceada": values["balanced_accuracy"],
                "PR AUC macro": values["macro_pr_auc"],
            }
            for node, values in selected["nodes"].items()
        ],
        hide_index=True,
        width="stretch",
        explanation=describe("performance", "Rendimiento por nodo"),
        key="research_ui_render_diagnostics5",
    )
    _interpretation(
        measure="Rendimiento de la misma semilla por nodo experimental.",
        direction="Costo bajo y métricas altas en todos los nodos.",
        result=(
            f"Se evaluaron {len(selected['nodes'])} particiones sobre sus "
            "ejemplos oficiales."
        ),
        conclusion=(
            "Las diferencias entre nodos señalan heterogeneidad que el promedio "
            "global oculta."
        ),
        limitation=(
            "Las particiones no representan contratistas reales; clases ausentes "
            "reducen el F1 computable."
        ),
    )


def _render_diagnostic_confusion(metrics: dict) -> None:
    labels = [f"Clase {index}" for index in range(5)]
    figure = go.Figure(
        go.Heatmap(
            z=metrics["confusion_matrix"],
            x=labels,
            y=labels,
            colorscale="YlOrBr",
            hovertemplate=(
                "Real: %{y}<br>Predicha: %{x}<br>Ejemplos: %{z}<extra></extra>"
            ),
        )
    )
    figure.update_layout(height=360, xaxis_title="Predicha", yaxis_title="Real")
    _plot(figure, describe("confusion", "Confusión en validación oficial"))
    render_table(
        [
            {
                "Clase real": index,
                "Ejemplos": metrics["support"][index],
                "Predicciones": metrics["predicted_support"][index],
                "Precisión": metrics["precision"][index],
                "Recall": metrics["recall"][index],
                "F1": metrics["f1"][index],
                "Costo medio de la clase": metrics["mean_cost_by_class"][index],
            }
            for index in range(5)
        ],
        hide_index=True,
        width="stretch",
        explanation=describe("class_metrics", "Diagnóstico por clase"),
        key="research_ui_render_diagnostic_confusion0",
    )
    _interpretation(
        measure=(
            "Errores entre clase real y predicha, y costo medio dentro de cada clase."
        ),
        direction="Más ejemplos en la diagonal y menor costo por clase son favorables.",
        result=(
            f"La clase 0 tiene recall {metrics['recall'][0]:.1%}; "
            f"la clase 4 tiene recall {metrics['recall'][4]:.1%}."
        ),
        conclusion=(
            "La matriz permite ver falsas alarmas y fallas de detección que "
            "ocultan los promedios."
        ),
        limitation="Las clases escasas producen estimaciones más inciertas.",
    )


def _render_training_history(report: dict | None) -> None:
    history = report.get("training_history") if report else None
    if not history:
        return
    st.caption("Convergencia de una ejecución representativa")
    figure = go.Figure()
    for regime in ("fedavg", "fedprox"):
        points = history.get(regime, [])
        figure.add_trace(
            go.Scatter(
                x=[point["round"] for point in points],
                y=[point["training_loss"] for point in points],
                mode="lines+markers",
                name=REGIME_LABELS[regime],
            )
        )
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Ronda federada",
        yaxis_title="Pérdida de entrenamiento",
        legend_orientation="h",
    )
    _plot(figure, describe("training", "Convergencia federada"))
    changes = [
        (
            REGIME_LABELS[regime],
            points[0]["training_loss"],
            points[-1]["training_loss"],
        )
        for regime in ("fedavg", "fedprox")
        if (points := history.get(regime, []))
    ]
    result = "; ".join(
        f"{regime}: {start:.4f} → {end:.4f}" for regime, start, end in changes
    )
    converged = all(end < start for _, start, end in changes)
    _interpretation(
        measure="Pérdida optimizada durante cada ronda federada.",
        direction="Una reducción estable indica convergencia del entrenamiento.",
        result=result,
        conclusion=(
            "Las curvas reducen su pérdida durante el ajuste."
            if converged
            else "Alguna curva no redujo su pérdida de inicio a fin."
        ),
        limitation="Una pérdida menor no garantiza menor costo en validación.",
    )


def _render_model_selection(path: Path) -> None:
    selection = _json(path / "selection.json")
    candidates = _csv(path / "candidates.csv")
    st.subheader("Selección del modelo e hiperparámetros")
    if selection is None or not candidates:
        st.info("Ejecuta la validación cruzada para seleccionar hiperparámetros.")
        return
    rows = [
        {
            "Candidato": row["candidate"],
            "Capas ocultas": row["hidden_sizes"],
            "Épocas": int(row["epochs"]),
            "Lote": int(row["batch_size"]),
            "Tasa de aprendizaje": float(row["learning_rate"]),
            "Regularización": float(row["weight_decay"]),
            "Parámetros": int(row["parameters"]),
            "Costo medio": float(row["mean_cost_mean"]),
            "DE costo": float(row["mean_cost_sd"]),
            "F1 macro": float(row["macro_f1_mean"]),
            "Seleccionado": row["selected"] == "True",
        }
        for row in candidates
    ]
    figure = go.Figure(
        go.Bar(
            x=[row["Candidato"] for row in rows],
            y=[row["Costo medio"] for row in rows],
            error_y={"type": "data", "array": [row["DE costo"] for row in rows]},
            marker_color=[
                "#f2b544" if row["Seleccionado"] else "#59616d" for row in rows
            ],
            hovertemplate="%{x}<br>Costo medio %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Costo medio de validación cruzada",
        showlegend=False,
    )
    _plot(figure, describe("selection", "Costo de candidatos con desviación estándar"))
    chosen = next(row for row in rows if row["Seleccionado"])
    _interpretation(
        measure="Costo medio y variación de cada hiperparámetro entre folds.",
        direction="Menor costo; los empates priorizan F1 macro y menor complejidad.",
        result=(
            f"{chosen['Candidato']} obtuvo {chosen['Costo medio']:.4f} ± "
            f"{chosen['DE costo']:.4f}."
        ),
        conclusion="La configuración resaltada minimizó el criterio principal.",
        limitation=(
            "La selección usa entrenamiento agrupado y no la validación oficial."
        ),
    )
    render_table(
        rows,
        hide_index=True,
        width="stretch",
        explanation=describe("selection", "Candidatos e hiperparámetros"),
        key="research_ui_render_model_selection0",
    )
    _interpretation(
        measure="Arquitectura, épocas, lote, tasa, regularización y parámetros.",
        direction="Se favorece el criterio de selección, no una complejidad mayor.",
        result=(
            f"La arquitectura elegida tiene capas {chosen['Capas ocultas']}, "
            f"{chosen['Parámetros']:,} parámetros y tasa "
            f"{chosen['Tasa de aprendizaje']}."
        ),
        conclusion="La misma arquitectura permite comparar los cuatro regímenes.",
        limitation="La búsqueda cubre solo los candidatos declarados.",
    )
    st.success(
        f"Configuración elegida: {selection['selected_candidate']}. "
        f"Criterio: {selection['selection_rule']}."
    )


def _render_cross_validation(path: Path) -> None:
    selection = _json(path / "selection.json")
    folds = _csv(path / "fold_metrics.csv")
    st.subheader("Validación cruzada agrupada")
    if selection is None or not folds:
        st.info("Ejecuta la selección de modelos para generar los folds.")
        return
    candidate = selection["selected_candidate"]
    selected = [row for row in folds if row["candidate"] == candidate]
    rows = [
        {
            "Fold": int(row["fold"]),
            "Vehículos": int(row["vehicles"]),
            "Costo medio": float(row["mean_cost"]),
            "F1 macro": float(row["macro_f1"]),
            "Exactitud balanceada": float(row["balanced_accuracy"]),
            "PR AUC macro": float(row["macro_pr_auc"]),
        }
        for row in selected
    ]
    figure = go.Figure(
        go.Bar(
            x=[f"Fold {row['Fold']}" for row in rows],
            y=[row["Costo medio"] for row in rows],
            marker_color="#f2b544",
            text=[f"{row['Costo medio']:.3f}" for row in rows],
            textposition="outside",
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Costo medio",
        showlegend=False,
    )
    _plot(figure, describe("folds", "Costo por fold"))
    costs = [row["Costo medio"] for row in rows]
    _interpretation(
        measure="Costo del candidato elegido en cada fold agrupado por vehículo.",
        direction=(
            "Menor costo y menor dispersión indican mejor generalización interna."
        ),
        result=(
            f"Promedio {sum(costs) / len(costs):.4f}; rango "
            f"{min(costs):.4f}–{max(costs):.4f}."
        ),
        conclusion="El rendimiento varía de forma acotada entre los cinco folds.",
        limitation=(
            "La validación cruzada permanece dentro del conjunto de entrenamiento."
        ),
    )
    render_table(
        rows,
        hide_index=True,
        width="stretch",
        explanation=describe("folds", "Resultados por fold"),
        key="research_ui_render_cross_validation0",
    )
    protocol = selection["cross_validation"]
    overlap = protocol["train_validation_vehicle_overlap"]
    _interpretation(
        measure="Integridad de los folds y ausencia de vehículos compartidos.",
        direction="El solapamiento debe ser cero.",
        result=(
            f"{protocol['folds']} folds, agrupación por {protocol['group']} y "
            f"solapamiento {overlap}."
        ),
        conclusion="No existe fuga del mismo vehículo entre ajuste y validación.",
        limitation=f"La estratificación aproxima {protocol['stratification']}.",
    )


def _render_statistics(path: Path) -> None:
    rows = _csv(path / "primary_statistics.csv")
    st.subheader("Pruebas estadísticas robustas")
    if not rows:
        st.info(
            "Ejecuta la evaluación final con varias semillas para generar inferencia pareada."
        )
        return
    display = [
        {
            "Referencia": row["reference"],
            "Candidato": row["candidate"],
            "Métrica": row["metric"],
            "Pares": 10,
            "Diferencia media": float(row["mean_difference"]),
            "IC 95 % inferior": float(row["ci_95_lower"]),
            "IC 95 % superior": float(row["ci_95_upper"]),
            "p ajustado Holm": float(row["p_value_holm"]),
            "Biserial de rangos": float(row["rank_biserial"]),
        }
        for row in rows
    ]
    robust = [
        row
        for row in display
        if row["p ajustado Holm"] < 0.05
        and (
            row["IC 95 % superior"] < 0
            if row["Métrica"]
            in {"mean_cost", "total_cost", "brier_score", "argmax_ece"}
            else row["IC 95 % inferior"] > 0
        )
    ]
    render_table(
        display,
        hide_index=True,
        width="stretch",
        explanation=describe("statistics", "Pruebas pareadas entre regímenes"),
        key="research_ui_render_statistics0",
    )
    _interpretation(
        measure=(
            "Mejora pareada, intervalo bootstrap, Wilcoxon con Holm y tamaño "
            "del efecto biserial."
        ),
        direction=(
            "Diferencia candidato menos referencia: negativa para costo, positiva "
            "para métricas de calidad. IC que excluya cero en esa dirección, "
            "p ajustado menor que 0.05 y efecto de magnitud relevante."
        ),
        result=(
            f"{len(robust)} de {len(display)} comparaciones cumplen IC y "
            "significancia simultáneamente."
        ),
        conclusion=(
            "Existen comparaciones con respaldo estadístico."
            if robust
            else "No existe evidencia suficiente para afirmar superioridad."
        ),
        limitation="Diez pares limitan la precisión y la potencia de las pruebas.",
    )
    if robust:
        st.success(
            f"{len(robust)} comparaciones muestran mejora con IC en la dirección favorable "
            "y p ajustado menor que 0.05."
        )
    else:
        st.warning(
            "Ninguna comparación cumple simultáneamente IC favorable y p ajustado "
            "menor que 0.05; no corresponde afirmar superioridad robusta."
        )


def _render_interpretability(path: Path) -> None:
    summary = _json(path / "summary.json")
    rows = _csv(path / "permutation_importance.csv")
    st.subheader("Importancia por permutación")
    if summary is None or not rows:
        st.info(
            "Ejecuta la importancia por permutación para explicar el modelo global."
        )
        return
    normalized = [
        {
            "Dimensión": row["dimension"],
            "Grupo": row["group"],
            "Variables": int(row["features"]),
            "Aumento del costo": float(row["mean_cost_increase"]),
            "DE": float(row["sd_cost_increase"]),
        }
        for row in rows
    ]
    source = sorted(
        (row for row in normalized if row["Dimensión"] == "source_variable"),
        key=lambda row: row["Aumento del costo"],
        reverse=True,
    )[:12]
    statistics = sorted(
        (row for row in normalized if row["Dimensión"] == "temporal_statistic"),
        key=lambda row: row["Aumento del costo"],
        reverse=True,
    )
    if not source or not statistics:
        st.warning("El artefacto de interpretabilidad está incompleto.")
        return
    figure = go.Figure(
        go.Bar(
            x=[row["Aumento del costo"] for row in reversed(source)],
            y=[row["Grupo"] for row in reversed(source)],
            orientation="h",
            marker_color="#f2b544",
            error_x={
                "type": "data",
                "array": [row["DE"] for row in reversed(source)],
            },
        )
    )
    figure.update_layout(
        height=420,
        xaxis_title="Aumento del costo medio al permutar",
        yaxis_title="Variable anónima de origen",
    )
    _plot(figure, describe("importance", "Importancia por variable de origen"))
    top = source[0]
    _interpretation(
        measure="Cambio del costo al romper la información de cada variable de origen.",
        direction="Un aumento positivo mayor indica más dependencia del modelo.",
        result=(
            f"{top['Grupo']} produjo el mayor aumento medio: "
            f"{top['Aumento del costo']:.4f}."
        ),
        conclusion=(
            "El modelo muestra dependencia positiva de ese grupo de señales anónimas."
            if top["Aumento del costo"] > 0
            else "Ningún grupo produjo un aumento positivo del costo."
        ),
        limitation=(
            "La permutación mide dependencia predictiva, no causalidad; variables "
            "correlacionadas pueden repartirse la importancia."
        ),
    )
    render_table(
        statistics,
        hide_index=True,
        width="stretch",
        explanation=describe("importance", "Importancia de estadísticos temporales"),
        key="research_ui_render_interpretability0",
    )
    best_statistic = statistics[0]
    _interpretation(
        measure="Dependencia del modelo respecto a cada resumen temporal.",
        direction="Mayor aumento del costo significa mayor aporte predictivo.",
        result=(
            f"{best_statistic['Grupo']} fue el estadístico más influyente con "
            f"{best_statistic['Aumento del costo']:.4f}."
        ),
        conclusion=(
            "Ese resumen temporal mostró el mayor aporte predictivo."
            if best_statistic["Aumento del costo"] > 0
            else "Ningún resumen temporal mostró importancia positiva."
        ),
        limitation=(
            f"Resultado del split {summary['split']}, régimen "
            f"{REGIME_LABELS.get(summary['regime'], summary['regime'])} y "
            f"{summary['config']['repeats']} permutaciones."
        ),
    )
    st.caption(
        f"SCANIA Component X · {summary['examples']:,} ejemplos · "
        f"Costo base {summary['baseline_mean_cost']:.4f} · {summary['model_id']}"
    )


def _render_sovereignty(path: Path) -> None:
    data = _json(path)
    st.subheader("Soberanía y transferencia")
    if data is None:
        st.info("Ejecuta la inferencia federada para mostrar la trazabilidad.")
        return
    transfer = data["transfer"]
    provenance = data.get("provenance", {})
    if provenance.get("verified"):
        st.caption(
            f"Datos {provenance['data_version'][:12]} · "
            f"entrenamiento {provenance['training_id'][:12]} · "
            "coincidencia de caché y modelos verificada"
        )
    else:
        st.warning(
            "El artefacto anterior no prueba la correspondencia entre caché, "
            "configuración y modelos. Regenera aprendizaje y publicación."
        )
    st.caption(
        "Nodos: particiones lógicas de un mismo dataset público. "
        "No representa un despliegue entre contratistas."
    )
    if data.get("centralized_training_uses_raw_data"):
        st.warning(
            "El régimen centralizado usa datos de entrenamiento reunidos; "
            "la afirmación de retención local no aplica a ese comparador."
        )
    columns = st.columns(4)
    columns[0].metric("Estados publicados", f"{data['coordinator_states']:,}")
    columns[1].metric("Variables crudas publicadas", data["coordinator_raw_features"])
    columns[2].metric("Registros crudos publicados", transfer["raw_records"])
    columns[3].metric("Volumen registrado", _bytes(transfer["payload_bytes"]))
    reduction = data.get("publication_reduction_ratio")
    if reduction is not None:
        st.metric(
            "Reducción frente a centralizar las variables de entrada",
            f"{reduction:.1%}",
            delta_color="off",
        )
    parameters = transfer["by_type"].get("model_parameters", {})
    if parameters.get("messages"):
        st.caption(
            f"{parameters['messages']} actualizaciones de parámetros · "
            f"{_bytes(parameters['payload_bytes'])} estimados para los nodos"
        )
    st.caption(
        "El coordinador de estados recibe riesgos y metadatos permitidos. "
        "Los bytes de estados se miden al serializar; los de parámetros se "
        "estiman a partir del tamaño del modelo."
    )
    rows = [
        {
            "Nodo": node,
            "Registros privados": values["private_records"],
            "Estados publicados": values["published_states"],
            "Cobertura de lecturas": values.get("mean_observation_quality"),
        }
        for node, values in data["nodes"].items()
    ]
    render_table(
        rows,
        hide_index=True,
        width="stretch",
        explanation=describe("federation", "Registros y estados por nodo"),
        key="research_ui_render_sovereignty0",
    )
    _interpretation(
        measure=(
            "Registros asignados, estados publicados y cobertura de lecturas "
            "por partición."
        ),
        direction="El estado publicado debe excluir registros y variables crudas.",
        result=(
            f"{sum(row['Registros privados'] for row in rows):,} registros privados; "
            f"{transfer['raw_records']} registros crudos transferidos."
        ),
        conclusion="Los estados publicados no contienen lecturas ni ventanas crudas.",
        limitation=(
            "La retención se evalúa en particiones lógicas de un solo proceso; "
            "no demuestra soberanía entre organizaciones independientes. La "
            "cobertura de lecturas no mide precisión predictiva."
        ),
    )


def _json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if amount < 1024 or unit == "GB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise RuntimeError("Unidad de almacenamiento no disponible.")


def _selection_path(root: Path) -> Path:
    final = root / "04_seleccion_modelo"
    return final if final.is_dir() else root / "model_selection"


def _explain(text: str) -> None:
    st.caption(f"Cómo interpretarlo · {text}")


def _interpretation(
    measure: str,
    direction: str,
    result: str,
    conclusion: str,
    limitation: str,
) -> None:
    with st.container(border=True):
        st.markdown(
            f"**Qué mide:** {measure}  \n"
            f"**Dirección favorable:** {direction}  \n"
            f"**Resultado:** {result}  \n"
            f"**Conclusión:** {conclusion}  \n"
            f"**Limitación:** {limitation}"
        )


def _plot(figure: go.Figure, explanation: Explanation) -> None:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#5f6b7a"},
        xaxis={"gridcolor": "#e1e6ec"},
        yaxis={"gridcolor": "#e1e6ec"},
    )
    render_plot(figure, explanation=explanation)
