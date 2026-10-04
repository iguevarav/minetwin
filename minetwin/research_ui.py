import csv
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from minetwin.paths import RESULTS_ROOT

REGIME_LABELS = {
    "local": "Local",
    "centralized": "Centralizado",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
}
POLICY_LABELS = {
    "p0_none": "P0 Â· Sin mantenimiento",
    "p1_threshold": "P1 Â· Umbral",
    "p2_local_predictive": "P2 Â· Predictiva local",
    "p3_coordinated": "P3 Â· Coordinada",
}


def render_learning_results(root: Path = RESULTS_ROOT) -> None:
    st.caption("VALIDACIÃ“N CON DATOS REALES")
    selection = _selection_path(root)
    verified_learning = _verified_learning_path(root)
    verified = verified_learning is not None
    final = verified_learning or root / "final_learning"
    diagnostics = root / (
        "final_diagnostics_verified" if verified else "final_diagnostics"
    )
    interpretability = root / (
        "final_interpretability_verified"
        if verified
        else "final_interpretability"
    )
    tabs = st.tabs(
        (
            "EDA",
            "Entrenamiento",
            "DiagnÃ³stico predictivo",
            "SelecciÃ³n e hiperparÃ¡metros",
            "ValidaciÃ³n cruzada",
            "Interpretabilidad",
            "EstadÃ­stica",
            "SoberanÃ­a",
        )
    )
    with tabs[0]:
        _render_dataset(root / "phase1_scania" / "summary.json")
        _render_eda(selection / "eda.json")
    with tabs[1]:
        if (final / "summary.json").is_file():
            _render_final_learning(final)
        else:
            _render_preliminary_learning(root / "phase2_models" / "metrics.json")
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
        if verified:
            federation_path = root / "final_federation_verified" / "summary.json"
        else:
            paths = (
                root / "final_federation" / "summary.json",
                root / "phase3_federation" / "summary.json",
            )
            federation_path = next(
                (path for path in paths if path.is_file()), paths[-1]
            )
        _render_sovereignty(federation_path)


def _verified_learning_path(root: Path) -> Path | None:
    path = root / "final_learning_verified"
    summary = _json(path / "summary.json")
    manifest = _json(path / "manifest.json")
    if (
        summary is not None
        and manifest is not None
        and manifest.get("status") == "complete"
        and summary.get("provenance", {}).get("verified")
    ):
        return path
    return None


def _render_dataset(path: Path) -> None:
    data = _json(path)
    if data is None:
        st.info("Ejecuta la preparaciÃ³n de SCANIA para mostrar el conjunto de datos.")
        return
    splits = {item["split"]: item for item in data["splits"]}
    columns = st.columns(4)
    columns[0].metric("Variables", data["features"])
    columns[1].metric("VehÃ­culos de entrenamiento", f"{splits['train']['vehicles']:,}")
    columns[2].metric(
        "VehÃ­culos de validaciÃ³n", f"{splits['validation']['vehicles']:,}"
    )
    columns[3].metric("Nodos", len(data["nodes"]))
    _interpretation(
        measure="TamaÃ±o y separaciÃ³n de los conjuntos reales usados por el estudio.",
        direction="No tiene direcciÃ³n Ã³ptima; verifica cobertura y ausencia de mezcla.",
        result=(
            f"{splits['train']['vehicles']:,} vehÃ­culos de entrenamiento y "
            f"{splits['validation']['vehicles']:,} de validaciÃ³n, con "
            f"{data['features']} variables."
        ),
        conclusion="Los conjuntos oficiales contienen vehÃ­culos distintos.",
        limitation="Las variables son anÃ³nimas y solo representan Component X.",
    )


def _render_eda(path: Path) -> None:
    data = _json(path)
    st.subheader("AnÃ¡lisis exploratorio")
    if data is None:
        st.info("Ejecuta la selecciÃ³n de modelos para generar el EDA reproducible.")
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
    _plot(figure)
    dominant_label, dominant_count = max(classes.items(), key=lambda item: item[1])
    _interpretation(
        measure="Cantidad de ejemplos causales disponibles para cada clase de riesgo.",
        direction="Una distribuciÃ³n mÃ¡s equilibrada reduce el sesgo hacia la mayorÃ­a.",
        result=(
            f"La clase {dominant_label} reÃºne {dominant_count:,} de "
            f"{train['examples']:,} ejemplos "
            f"({dominant_count / train['examples']:.1%})."
        ),
        conclusion=(
            "El desbalance exige costo, F1 macro y exactitud balanceada; la exactitud "
            "simple puede ocultar errores crÃ­ticos."
        ),
        limitation="La distribuciÃ³n describe la muestra y no el riesgo poblacional.",
    )
    distribution = [
        {
            "Nodo": node,
            "Ejemplos": count,
            "ParticipaciÃ³n": count / train["examples"],
        }
        for node, count in train["nodes"].items()
    ]
    st.dataframe(distribution, hide_index=True, width="stretch")
    largest = max(distribution, key=lambda row: row["Ejemplos"])
    smallest = min(distribution, key=lambda row: row["Ejemplos"])
    _interpretation(
        measure="Cantidad y participaciÃ³n de ejemplos privados por nodo federado.",
        direction=(
            "TamaÃ±os similares reducen la influencia desproporcionada de un nodo."
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
        st.dataframe(
            [
                {"Conjunto": split, "ProporciÃ³n ausente": value}
                for split, value in missing.items()
            ],
            hide_index=True,
            width="stretch",
        )
        worst_split, worst_rate = max(missing.items(), key=lambda item: item[1])
        _interpretation(
            measure="ProporciÃ³n de celdas ausentes antes de la imputaciÃ³n causal.",
            direction="Una proporciÃ³n menor conserva mÃ¡s evidencia observada.",
            result=f"El mÃ¡ximo corresponde a {worst_split}: {worst_rate:.2%}.",
            conclusion=(
                "La ausencia se conserva como indicador y no se trata como cero."
            ),
            limitation=(
                "La tasa global puede ocultar concentraciÃ³n en variables concretas."
            ),
        )
    separation = [
        {
            "Variable derivada": item["feature"],
            "SeparaciÃ³n entre clases Î·Â²": item["eta_squared"],
        }
        for item in data["top_class_separation_features"]
    ]
    st.dataframe(
        separation,
        hide_index=True,
        width="stretch",
    )
    strongest = separation[0]
    _interpretation(
        measure="ProporciÃ³n de variaciÃ³n asociada descriptivamente con la clase.",
        direction="Un Î·Â² mayor indica mÃ¡s separaciÃ³n entre clases.",
        result=(
            f"{strongest['Variable derivada']} obtuvo el mayor Î·Â²: "
            f"{strongest['SeparaciÃ³n entre clases Î·Â²']:.4f}."
        ),
        conclusion="La variable encabeza la separaciÃ³n descriptiva del conjunto.",
        limitation=(
            "Î·Â² no demuestra causalidad ni importancia para el modelo entrenado."
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
            "RÃ©gimen": REGIME_LABELS.get(regime, regime),
            "Costo": values["global"]["total_cost"],
            "F1 macro": values["global"]["macro_f1"],
            "Exactitud balanceada": values["global"]["balanced_accuracy"],
            "PR AUC macro": values["global"]["macro_pr_auc"],
        }
        for regime, values in report["results"].items()
    ]
    st.caption(
        "Resultado preliminar de una semilla; no representa inferencia estadÃ­stica."
    )
    _learning_chart(rows)
    best_f1 = max(rows, key=lambda row: row["F1 macro"])
    _interpretation(
        measure="F1 macro, promedio del F1 de las cinco clases con igual peso.",
        direction="Un valor mÃ¡s alto representa mejor equilibrio entre clases.",
        result=(
            f"{best_f1['RÃ©gimen']} obtuvo el mayor F1 macro: "
            f"{best_f1['F1 macro']:.4f}."
        ),
        conclusion="Ese rÃ©gimen mostrÃ³ el mejor equilibrio en esta ejecuciÃ³n.",
        limitation="Una sola semilla no permite inferencia estadÃ­stica.",
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    best_cost = min(rows, key=lambda row: row["Costo"])
    _interpretation(
        measure="Costo total segÃºn la matriz oficial de errores SCANIA.",
        direction="Un costo menor penaliza menos los errores crÃ­ticos.",
        result=(
            f"{best_cost['RÃ©gimen']} obtuvo el menor costo: "
            f"{best_cost['Costo']:.2f}."
        ),
        conclusion="Es el mejor resultado operativo observado en esta semilla.",
        limitation="Debe confirmarse con semillas pareadas e incertidumbre.",
    )
    _render_training_history(report)


def _render_final_learning(path: Path) -> None:
    summary = _json(path / "summary.json")
    rows = [
        {
            "RÃ©gimen": REGIME_LABELS.get(regime, regime),
            "Costo": metrics["total_cost"]["mean"],
            "F1 macro": metrics["macro_f1"]["mean"],
            "Exactitud balanceada": metrics["balanced_accuracy"]["mean"],
            "PR AUC macro": metrics["macro_pr_auc"]["mean"],
        }
        for regime, metrics in summary["aggregates"].items()
    ]
    st.subheader("Rendimiento de los modelos")
    st.caption(f"{summary['seeds']} semillas de evaluaciÃ³n pareadas.")
    _learning_chart(rows)
    best_f1 = max(rows, key=lambda row: row["F1 macro"])
    _interpretation(
        measure="F1 macro medio sobre las mismas semillas y validaciÃ³n oficial.",
        direction="Un valor mayor indica mejor equilibrio entre las cinco clases.",
        result=f"{best_f1['RÃ©gimen']} alcanzÃ³ {best_f1['F1 macro']:.4f}.",
        conclusion="Fue el mayor F1 macro medio observado entre los regÃ­menes.",
        limitation="La superioridad requiere intervalo y valor p ajustado.",
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    best = min(rows, key=lambda row: row["Costo"])
    st.caption(
        f"Menor costo medio observado: {best['RÃ©gimen']} ({best['Costo']:.2f}). "
        "Revisar baselines, errores por clase y pruebas pareadas antes de "
        "concluir utilidad."
    )
    _interpretation(
        measure="Costo, F1 macro, exactitud balanceada y PR AUC medios.",
        direction="Menor costo y mayores mÃ©tricas restantes son favorables.",
        result=f"{best['RÃ©gimen']} presentÃ³ el menor costo medio: {best['Costo']:.2f}.",
        conclusion="Es el mejor costo observado sobre la validaciÃ³n oficial.",
        limitation=(
            "Las mÃ©tricas son bajas y deben leerse junto con la inferencia estadÃ­stica."
        ),
    )
    artifacts = sorted(path.glob("seed_*/metrics.json"))
    if artifacts:
        _render_training_history(_json(artifacts[0]))


def _learning_chart(rows: list[dict]) -> None:
    figure = go.Figure(
        go.Bar(
            x=[row["RÃ©gimen"] for row in rows],
            y=[row["F1 macro"] for row in rows],
            marker_color="#f2b544",
            hovertemplate="%{x}<br>F1 macro %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=300,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="F1 macro",
        showlegend=False,
    )
    _plot(figure)


def _render_diagnostics(path: Path) -> None:
    report = _json(path)
    st.subheader("DiagnÃ³stico predictivo")
    if report is None:
        st.info(
            "Genera el diagnÃ³stico sobre una semilla entrenada para ver "
            "baselines, calibraciÃ³n y errores."
        )
        return
    st.caption(
        f"ValidaciÃ³n oficial SCANIA Â· semilla {report['training_config']['seed']}. "
        "DiagnÃ³stico descriptivo. "
        "La selecciÃ³n de hiperparÃ¡metros se hizo solo con entrenamiento."
    )
    train = report["training_class_counts"]
    validation = report["validation_class_counts"]
    st.dataframe(
        [
            {
                "Clase": index,
                "Entrenamiento": train[index],
                "ProporciÃ³n entrenamiento": train[index] / sum(train),
                "ValidaciÃ³n": validation[index],
                "ProporciÃ³n validaciÃ³n": validation[index] / sum(validation),
            }
            for index in range(5)
        ],
        hide_index=True,
        width="stretch",
    )
    _interpretation(
        measure=(
            "Frecuencia relativa de cada clase antes y despuÃ©s de la "
            "separaciÃ³n oficial."
        ),
        direction="Las proporciones similares facilitan transportar el modelo.",
        result=(
            f"La clase 0 pasa de {train[0] / sum(train):.1%} en entrenamiento "
            f"a {validation[0] / sum(validation):.1%} en validaciÃ³n."
        ),
        conclusion=(
            "Existe un cambio de distribuciÃ³n que afecta la interpretaciÃ³n "
            "de las mÃ©tricas."
        ),
        limitation="La diferencia descriptiva no identifica por sÃ­ sola su causa.",
    )
    baseline_labels = {
        "always_0": "Siempre clase 0",
        "always_4": "Siempre clase 4",
        "train_prior": "Prior de entrenamiento",
    }
    baselines = report["baselines"]
    st.dataframe(
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
    )
    best_baseline = min(
        baselines, key=lambda key: baselines[key]["global"]["mean_cost"]
    )
    _interpretation(
        measure="Costo y clasificaciÃ³n de tres reglas sin entrenamiento de red.",
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
            "Los baselines se definieron sin elegir una regla sobre la validaciÃ³n."
        ),
    )
    regime = st.selectbox(
        "RÃ©gimen para examinar",
        tuple(report["results"]),
        format_func=lambda value: REGIME_LABELS.get(value, value),
        key="diagnostic_regime",
    )
    selected = report["results"][regime]
    global_metrics = selected["global"]
    st.dataframe(
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
    )
    argmax = report["argmax_results"][regime]
    st.dataframe(
        [
            {
                "DecisiÃ³n": name,
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
    )
    _interpretation(
        measure="Dos reglas de decisiÃ³n aplicadas a las mismas probabilidades.",
        direction="Menor costo y mayor equilibrio entre clases son favorables.",
        result=(
            f"Costo medio: matriz {global_metrics['mean_cost']:.3f}; "
            f"mÃ¡xima probabilidad {argmax['mean_cost']:.3f}."
        ),
        conclusion=(
            "La matriz penaliza mÃ¡s los errores sobre clases prÃ³ximas a reparaciÃ³n."
        ),
        limitation=(
            "Esta comparaciÃ³n es descriptiva; no selecciona una regla con "
            "validaciÃ³n oficial."
        ),
    )
    baseline_cost = baselines[best_baseline]["global"]["mean_cost"]
    baseline_f1 = max(item["global"]["macro_f1"] for item in baselines.values())
    if (
        global_metrics["mean_cost"] >= baseline_cost
        or global_metrics["macro_f1"] <= baseline_f1
    ):
        st.warning(
            "Este resultado no supera simultÃ¡neamente el costo del mejor baseline "
            "y el F1 macro basal. No demuestra calidad prÃ¡ctica para mantenimiento."
        )
    else:
        st.info(
            "Supera las referencias triviales en esta semilla; falta validar "
            "estabilidad estadÃ­stica y utilidad en una operaciÃ³n minera."
        )
    _interpretation(
        measure=(
            "Costo de errores, calidad por clase, precisiÃ³n-recall y calibraciÃ³n "
            "de las probabilidades sin recalibraciÃ³n posterior."
        ),
        direction="Menor costo, Brier y ECE; mayor F1, exactitud balanceada y PR AUC.",
        result=(
            f"{REGIME_LABELS[regime]}: costo {global_metrics['mean_cost']:.3f}, "
            f"F1 {global_metrics['macro_f1']:.3f}, "
            f"Brier {global_metrics['brier_score']:.3f}, "
            f"ECE {global_metrics['argmax_ece']:.3f}."
        ),
        conclusion=(
            "Una probabilidad confiable requiere Brier y ECE bajos ademÃ¡s de "
            "buena clasificaciÃ³n."
        ),
        limitation=(
            "ECE agrupa la confianza de la clase mÃ¡s probable en diez intervalos; "
            "no mide la calibraciÃ³n de la decisiÃ³n por costo. Esta semilla no "
            "permite inferencia."
        ),
    )
    st.dataframe(
        [
            {
                "Confianza": f"{item['lower']:.0%}â€“{item['upper']:.0%}",
                "Ejemplos": item["examples"],
                "Confianza media": item["mean_confidence"],
                "Exactitud top-1": item["accuracy"],
            }
            for item in global_metrics["calibration_bins"]
            if item["examples"]
        ],
        hide_index=True,
        width="stretch",
    )
    _interpretation(
        measure="Exactitud observada frente a confianza media por intervalo.",
        direction="Ambas cifras deben ser cercanas en cada intervalo.",
        result=f"ECE top-1: {global_metrics['argmax_ece']:.3f}.",
        conclusion=(
            "Los intervalos muestran dÃ³nde se concentra el error de calibraciÃ³n."
        ),
        limitation=(
            "La exactitud top-1 difiere de la clase elegida por la matriz de costo."
        ),
    )
    _render_diagnostic_confusion(global_metrics)
    st.dataframe(
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
    )
    _interpretation(
        measure="Rendimiento de la misma semilla por nodo experimental.",
        direction="Costo bajo y mÃ©tricas altas en todos los nodos.",
        result=(
            f"Se evaluaron {len(selected['nodes'])} particiones sobre sus "
            "ejemplos oficiales."
        ),
        conclusion=(
            "Las diferencias entre nodos seÃ±alan heterogeneidad que el promedio "
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
    _plot(figure)
    st.dataframe(
        [
            {
                "Clase real": index,
                "Ejemplos": metrics["support"][index],
                "Predicciones": metrics["predicted_support"][index],
                "PrecisiÃ³n": metrics["precision"][index],
                "Recall": metrics["recall"][index],
                "F1": metrics["f1"][index],
                "Costo medio de la clase": metrics["mean_cost_by_class"][index],
            }
            for index in range(5)
        ],
        hide_index=True,
        width="stretch",
    )
    _interpretation(
        measure=(
            "Errores entre clase real y predicha, y costo medio dentro de cada clase."
        ),
        direction="MÃ¡s ejemplos en la diagonal y menor costo por clase son favorables.",
        result=(
            f"La clase 0 tiene recall {metrics['recall'][0]:.1%}; "
            f"la clase 4 tiene recall {metrics['recall'][4]:.1%}."
        ),
        conclusion=(
            "La matriz permite ver falsas alarmas y fallas de detecciÃ³n que "
            "ocultan los promedios."
        ),
        limitation="Las clases escasas producen estimaciones mÃ¡s inciertas.",
    )


def _render_training_history(report: dict | None) -> None:
    history = report.get("training_history") if report else None
    if not history:
        return
    st.caption("Convergencia de una ejecuciÃ³n representativa")
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
        yaxis_title="PÃ©rdida de entrenamiento",
        legend_orientation="h",
    )
    _plot(figure)
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
        f"{regime}: {start:.4f} â†’ {end:.4f}"
        for regime, start, end in changes
    )
    converged = all(end < start for _, start, end in changes)
    _interpretation(
        measure="PÃ©rdida optimizada durante cada ronda federada.",
        direction="Una reducciÃ³n estable indica convergencia del entrenamiento.",
        result=result,
        conclusion=(
            "Las curvas reducen su pÃ©rdida durante el ajuste."
            if converged
            else "Alguna curva no redujo su pÃ©rdida de inicio a fin."
        ),
        limitation="Una pÃ©rdida menor no garantiza menor costo en validaciÃ³n.",
    )


def _render_model_selection(path: Path) -> None:
    selection = _json(path / "selection.json")
    candidates = _csv(path / "candidates.csv")
    st.subheader("SelecciÃ³n del modelo e hiperparÃ¡metros")
    if selection is None or not candidates:
        st.info("Ejecuta la validaciÃ³n cruzada para seleccionar hiperparÃ¡metros.")
        return
    rows = [
        {
            "Candidato": row["candidate"],
            "Capas ocultas": row["hidden_sizes"],
            "Ã‰pocas": int(row["epochs"]),
            "Lote": int(row["batch_size"]),
            "Tasa de aprendizaje": float(row["learning_rate"]),
            "RegularizaciÃ³n": float(row["weight_decay"]),
            "ParÃ¡metros": int(row["parameters"]),
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
        yaxis_title="Costo medio de validaciÃ³n cruzada",
        showlegend=False,
    )
    _plot(figure)
    chosen = next(row for row in rows if row["Seleccionado"])
    _interpretation(
        measure="Costo medio y variaciÃ³n de cada hiperparÃ¡metro entre folds.",
        direction="Menor costo; los empates priorizan F1 macro y menor complejidad.",
        result=(
            f"{chosen['Candidato']} obtuvo {chosen['Costo medio']:.4f} Â± "
            f"{chosen['DE costo']:.4f}."
        ),
        conclusion="La configuraciÃ³n resaltada minimizÃ³ el criterio principal.",
        limitation=(
            "La selecciÃ³n usa entrenamiento agrupado y no la validaciÃ³n oficial."
        ),
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    _interpretation(
        measure="Arquitectura, Ã©pocas, lote, tasa, regularizaciÃ³n y parÃ¡metros.",
        direction="Se favorece el criterio de selecciÃ³n, no una complejidad mayor.",
        result=(
            f"La arquitectura elegida tiene capas {chosen['Capas ocultas']}, "
            f"{chosen['ParÃ¡metros']:,} parÃ¡metros y tasa "
            f"{chosen['Tasa de aprendizaje']}."
        ),
        conclusion="La misma arquitectura permite comparar los cuatro regÃ­menes.",
        limitation="La bÃºsqueda cubre solo los candidatos declarados.",
    )
    st.success(
        f"ConfiguraciÃ³n elegida: {selection['selected_candidate']}. "
        f"Criterio: {selection['selection_rule']}."
    )


def _render_cross_validation(path: Path) -> None:
    selection = _json(path / "selection.json")
    folds = _csv(path / "fold_metrics.csv")
    st.subheader("ValidaciÃ³n cruzada agrupada")
    if selection is None or not folds:
        st.info("Ejecuta la selecciÃ³n de modelos para generar los folds.")
        return
    candidate = selection["selected_candidate"]
    selected = [row for row in folds if row["candidate"] == candidate]
    rows = [
        {
            "Fold": int(row["fold"]),
            "VehÃ­culos": int(row["vehicles"]),
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
    _plot(figure)
    costs = [row["Costo medio"] for row in rows]
    _interpretation(
        measure="Costo del candidato elegido en cada fold agrupado por vehÃ­culo.",
        direction=(
            "Menor costo y menor dispersiÃ³n indican mejor generalizaciÃ³n interna."
        ),
        result=(
            f"Promedio {sum(costs) / len(costs):.4f}; rango "
            f"{min(costs):.4f}â€“{max(costs):.4f}."
        ),
        conclusion="El rendimiento varÃ­a de forma acotada entre los cinco folds.",
        limitation=(
            "La validaciÃ³n cruzada permanece dentro del conjunto de entrenamiento."
        ),
    )
    st.dataframe(rows, hide_index=True, width="stretch")
    protocol = selection["cross_validation"]
    overlap = protocol["train_validation_vehicle_overlap"]
    _interpretation(
        measure="Integridad de los folds y ausencia de vehÃ­culos compartidos.",
        direction="El solapamiento debe ser cero.",
        result=(
            f"{protocol['folds']} folds, agrupaciÃ³n por {protocol['group']} y "
            f"solapamiento {overlap}."
        ),
        conclusion="No existe fuga del mismo vehÃ­culo entre ajuste y validaciÃ³n.",
        limitation=f"La estratificaciÃ³n aproxima {protocol['stratification']}.",
    )


def _render_statistics(path: Path) -> None:
    rows = _csv(path / "statistics.csv")
    st.subheader("Pruebas estadÃ­sticas robustas")
    if not rows:
        st.info(
            "Ejecuta la evaluaciÃ³n final con varias semillas para generar inferencia pareada."
        )
        return
    display = [
        {
            "Referencia": row["reference"],
            "Candidato": row["candidate"],
            "MÃ©trica": row["metric"],
            "Pares": int(row["pairs"]),
            "Mejora media": float(row["mean_improvement"]),
            "IC 95 % inferior": float(row["improvement_ci_lower"]),
            "IC 95 % superior": float(row["improvement_ci_upper"]),
            "p ajustado Holm": float(row["p_value_holm"]),
            "Biserial de rangos": float(row["rank_biserial"]),
        }
        for row in rows
    ]
    robust = [
        row
        for row in display
        if row["IC 95 % inferior"] > 0 and row["p ajustado Holm"] < 0.05
    ]
    st.dataframe(display, hide_index=True, width="stretch")
    _interpretation(
        measure=(
            "Mejora pareada, intervalo bootstrap, Wilcoxon con Holm y tamaÃ±o "
            "del efecto biserial."
        ),
        direction=(
            "Mejora positiva, IC sobre cero, p ajustado menor que 0.05 y efecto "
            "de magnitud relevante."
        ),
        result=(
            f"{len(robust)} de {len(display)} comparaciones cumplen IC y "
            "significancia simultÃ¡neamente."
        ),
        conclusion=(
            "Existen comparaciones con respaldo estadÃ­stico."
            if robust
            else "No existe evidencia suficiente para afirmar superioridad."
        ),
        limitation="Diez pares limitan la precisiÃ³n y la potencia de las pruebas.",
    )
    if robust:
        st.success(
            f"{len(robust)} comparaciones muestran mejora positiva con IC sobre cero "
            "y p ajustado menor que 0.05."
        )
    else:
        st.warning(
            "Ninguna comparaciÃ³n cumple simultÃ¡neamente IC sobre cero y p ajustado "
            "menor que 0.05; no corresponde afirmar superioridad robusta."
        )


def _render_interpretability(path: Path) -> None:
    summary = _json(path / "summary.json")
    rows = _csv(path / "permutation_importance.csv")
    st.subheader("Importancia por permutaciÃ³n")
    if summary is None or not rows:
        st.info(
            "Ejecuta la importancia por permutaciÃ³n para explicar el modelo global."
        )
        return
    normalized = [
        {
            "DimensiÃ³n": row["dimension"],
            "Grupo": row["group"],
            "Variables": int(row["features"]),
            "Aumento del costo": float(row["mean_cost_increase"]),
            "DE": float(row["sd_cost_increase"]),
        }
        for row in rows
    ]
    source = sorted(
        (row for row in normalized if row["DimensiÃ³n"] == "source_variable"),
        key=lambda row: row["Aumento del costo"],
        reverse=True,
    )[:12]
    statistics = sorted(
        (
            row
            for row in normalized
            if row["DimensiÃ³n"] == "temporal_statistic"
        ),
        key=lambda row: row["Aumento del costo"],
        reverse=True,
    )
    if not source or not statistics:
        st.warning("El artefacto de interpretabilidad estÃ¡ incompleto.")
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
        yaxis_title="Variable anÃ³nima de origen",
    )
    _plot(figure)
    top = source[0]
    _interpretation(
        measure="Cambio del costo al romper la informaciÃ³n de cada variable de origen.",
        direction="Un aumento positivo mayor indica mÃ¡s dependencia del modelo.",
        result=(
            f"{top['Grupo']} produjo el mayor aumento medio: "
            f"{top['Aumento del costo']:.4f}."
        ),
        conclusion=(
            "El modelo muestra dependencia positiva de ese grupo de seÃ±ales anÃ³nimas."
            if top["Aumento del costo"] > 0
            else "NingÃºn grupo produjo un aumento positivo del costo."
        ),
        limitation=(
            "La permutaciÃ³n mide dependencia predictiva, no causalidad; variables "
            "correlacionadas pueden repartirse la importancia."
        ),
    )
    st.dataframe(statistics, hide_index=True, width="stretch")
    best_statistic = statistics[0]
    _interpretation(
        measure="Dependencia del modelo respecto a cada resumen temporal.",
        direction="Mayor aumento del costo significa mayor aporte predictivo.",
        result=(
            f"{best_statistic['Grupo']} fue el estadÃ­stico mÃ¡s influyente con "
            f"{best_statistic['Aumento del costo']:.4f}."
        ),
        conclusion=(
            "Ese resumen temporal mostrÃ³ el mayor aporte predictivo."
            if best_statistic["Aumento del costo"] > 0
            else "NingÃºn resumen temporal mostrÃ³ importancia positiva."
        ),
        limitation=(
            f"Resultado del split {summary['split']}, rÃ©gimen "
            f"{REGIME_LABELS.get(summary['regime'], summary['regime'])} y "
            f"{summary['config']['repeats']} permutaciones."
        ),
    )
    st.caption(
        f"SCANIA Component X Â· {summary['examples']:,} ejemplos Â· "
        f"Costo base {summary['baseline_mean_cost']:.4f} Â· {summary['model_id']}"
    )


def _render_sovereignty(path: Path) -> None:
    data = _json(path)
    st.subheader("SoberanÃ­a y transferencia")
    if data is None:
        st.info("Ejecuta la inferencia federada para mostrar la trazabilidad.")
        return
    transfer = data["transfer"]
    provenance = data.get("provenance", {})
    if provenance.get("verified"):
        st.caption(
            f"Datos {provenance['data_version'][:12]} Â· "
            f"entrenamiento {provenance['training_id'][:12]} Â· "
            "coincidencia de cachÃ© y modelos verificada"
        )
    else:
        st.warning(
            "El artefacto anterior no prueba la correspondencia entre cachÃ©, "
            "configuraciÃ³n y modelos. Regenera aprendizaje y publicaciÃ³n."
        )
    st.caption(
        "Nodos: particiones lÃ³gicas de un mismo dataset pÃºblico. "
        "No representa un despliegue entre contratistas."
    )
    if data.get("centralized_training_uses_raw_data"):
        st.warning(
            "El rÃ©gimen centralizado usa datos de entrenamiento reunidos; "
            "la afirmaciÃ³n de retenciÃ³n local no aplica a ese comparador."
        )
    columns = st.columns(4)
    columns[0].metric("Estados publicados", f"{data['coordinator_states']:,}")
    columns[1].metric("Variables crudas publicadas", data["coordinator_raw_features"])
    columns[2].metric("Registros crudos publicados", transfer["raw_records"])
    columns[3].metric("Volumen registrado", _bytes(transfer["payload_bytes"]))
    reduction = data.get("publication_reduction_ratio")
    if reduction is not None:
        st.metric(
            "ReducciÃ³n frente a centralizar las variables de entrada",
            f"{reduction:.1%}",
            delta_color="off",
        )
    parameters = transfer["by_type"].get("model_parameters", {})
    if parameters.get("messages"):
        st.caption(
            f"{parameters['messages']} actualizaciones de parÃ¡metros Â· "
            f"{_bytes(parameters['payload_bytes'])} estimados para los nodos"
        )
    st.caption(
        "El coordinador de estados recibe riesgos y metadatos permitidos. "
        "Los bytes de estados se miden al serializar; los de parÃ¡metros se "
        "estiman a partir del tamaÃ±o del modelo."
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
    st.dataframe(rows, hide_index=True, width="stretch")
    _interpretation(
        measure=(
            "Registros asignados, estados publicados y cobertura de lecturas "
            "por particiÃ³n."
        ),
        direction="El estado publicado debe excluir registros y variables crudas.",
        result=(
            f"{sum(row['Registros privados'] for row in rows):,} registros privados; "
            f"{transfer['raw_records']} registros crudos transferidos."
        ),
        conclusion="Los estados publicados no contienen lecturas ni ventanas crudas.",
        limitation=(
            "La retenciÃ³n se evalÃºa en particiones lÃ³gicas de un solo proceso; "
            "no demuestra soberanÃ­a entre organizaciones independientes. La "
            "cobertura de lecturas no mide precisiÃ³n predictiva."
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
    final = root / "final_selection"
    return final if final.is_dir() else root / "model_selection"


def _explain(text: str) -> None:
    st.caption(f"CÃ³mo interpretarlo Â· {text}")


def _interpretation(
    measure: str,
    direction: str,
    result: str,
    conclusion: str,
    limitation: str,
) -> None:
    with st.container(border=True):
        st.markdown(
            f"**QuÃ© mide:** {measure}  \n"
            f"**DirecciÃ³n favorable:** {direction}  \n"
            f"**Resultado:** {result}  \n"
            f"**ConclusiÃ³n:** {conclusion}  \n"
            f"**LimitaciÃ³n:** {limitation}"
        )


def _plot(figure: go.Figure) -> None:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#5f6b7a"},
        xaxis={"gridcolor": "#e1e6ec"},
        yaxis={"gridcolor": "#e1e6ec"},
    )
    st.plotly_chart(
        figure,
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
    )
