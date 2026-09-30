# MineTwin: gemelo digital federado para mantenimiento predictivo

MineTwin es un prototipo académico reproducible para estudiar predicción de riesgo de fallo y coordinación de mantenimiento. Distingue dos líneas de evidencia: aprendizaje supervisado/federado sobre **SCANIA Component X** (variables físicas anónimas) y simulación de una flota minera ficticia. SCANIA sustenta la evaluación predictiva; la simulación sustenta políticas operativas. Los hallazgos no son directamente generalizables a flotas mineras reales sin validación externa.

## 1. Instrucciones de reproducibilidad

### Requisitos e instalación

Se requiere **Python 3.12.x**. Las versiones están fijadas en `pyproject.toml`: NumPy 1.26.4, Plotly 6.4.0, Pydantic 2.13.5 y Streamlit 1.63.0. El extra `learning` incorpora PyTorch 2.14.0; `dev`, Pytest 9.0.1 y Ruff 0.16.5.

```powershell
git clone <URL_DEL_REPOSITORIO>
Set-Location minetwin
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[learning,dev]"
python -m pytest -q -p no:cacheprovider
python -m ruff check .
```

### Ejecución de Streamlit

```powershell
python -m streamlit run app.py
```

El modo de simulación se ejecuta sin datos externos. El modo **Datos reales · SCANIA** requiere los artefactos de preparación siguientes.

### Preparación reproducible de SCANIA Component X

El conjunto público no se versiona en este repositorio. Descárguelo desde la fuente oficial, respetando su licencia, y ubíquelo en `minetwin/data/scania/` sin alterar los nombres: los tres `*_operational_readouts.csv`, los tres `*_specifications.csv`, `train_tte.csv`, `validation_labels.csv` y `test_labels.csv`. Cada directorio de salida debe ser nuevo para impedir la sobrescritura de evidencia experimental.

```powershell
$dataset = "minetwin\data\scania"
python -m minetwin --prepare-scania $dataset --output results\phase1_scania
python -m minetwin --prepare-scania-replay $dataset --phase1 results\phase1_scania --output results\phase1_replay
python -m minetwin.learning prepare --dataset $dataset --phase1 results\phase1_scania --output results\phase2_cache --window-size 12 --stride 1
python -m minetwin.learning select --cache results\phase2_cache --output results\model_selection --folds 5 --seed 42
python -m minetwin.learning train --cache results\phase2_cache --output results\phase2_models --epochs 20 --rounds 10 --local-epochs 1 --batch-size 256 --seed 42
python -m minetwin.learning publish --cache results\phase2_cache --models results\phase2_models --output results\phase3_federation --split validation --regime fedprox
python -m minetwin.learning interpret --cache results\phase2_cache --models results\phase2_models --output results\interpretability --split validation --regime fedprox --repeats 3 --seed 42
python minetwin\study.py learning --cache results\phase2_cache --selection results\model_selection --output results\final_learning --seeds 100,101,102,103,104,105,106,107,108,109
```

La evaluación de políticas simuladas se reproduce de forma independiente:

```powershell
python -m minetwin --compare --steps 450 --scenario engine_degradation --output results\comparison_run
python -m minetwin --evaluate-fleet --dataset-role evaluation --output results\fleet_evaluation
```

Los artefactos de trazabilidad son JSON, CSV, NPZ, modelos `.pt` y HTML; `selection.json`, `candidates.csv`, `fold_metrics.csv`, `metrics.json`, `summary.json` y `statistics.csv` contienen las salidas cuantitativas auditables.

## 2. Metodología bajo CRISP-DM

### 2.1 Comprensión del problema y datos

El objetivo es clasificar la inminencia de reparación del componente X en cinco clases: 0, más de 48 pasos o sin reparación; 1, 24--48; 2, 12--24; 3, 6--12; y 4, 0--6 pasos. La decisión minimiza coste esperado con la matriz SCANIA, que penaliza especialmente subestimar el riesgo (por ejemplo, clase real 4 predicha como 0: coste 500; el error inverso: 10).

Los *splits* oficiales `train`, `validation` y `test` se conservan. El cargador verifica esquema, identidades, contigüidad y orden temporal; también rechaza vehículos compartidos entre *splits*. Tres nodos experimentales (`alpha`, `beta`, `gamma`) se asignan determinísticamente desde especificaciones categóricas (concentración 0.3, semilla 42); no representan contratistas reales.

### 2.2 EDA (análisis exploratorio de datos)

| Dimensión | Implementación extraída del código |
|---|---|
| Estructura y calidad | Conteos de vehículos, lecturas, variables, especificaciones y resultados por *split*; validación de esquema, identidad y secuencia temporal. |
| Ausencia | Proporción de valores ausentes por *split* antes de imputación; los ausentes nunca se sustituyen por cero. |
| Desbalance | Frecuencias de las cinco clases globales y por nodo; se advierte que la exactitud simple es insuficiente. |
| Heterogeneidad | Distribución de vehículos y ejemplos causales entre nodos federados. |
| Separación descriptiva | Doce variables derivadas con mayor η² entre clases (varianza entre clases/varianza total); no se interpreta como causalidad. |
| Visualización | Barras de clase, vehículos por nodo y ausencia; tablas de clases, nodos, ausencia y η²; telemetría como series temporales por vehículo. |

La ingeniería de variables utiliza ventanas causales de hasta 12 lecturas y paso 1. Para cada señal se derivan último valor, media, desviación estándar, tendencia entre extremos y tasa de ausencia; además se añaden `log1p` del tamaño y duración de ventana. Los cuatro primeros resúmenes se transforman como `sign(x)·log(1+|x|)`. La imputación es *last observation carried forward* causal con indicadores de ausencia y sin cruzar vehículos. En entrenamiento se conserva la última ventana causal de cada clase por vehículo; en validación/prueba, la última ventana disponible.

### 2.3 Entrenamiento

Se entrenan MLP de cinco salidas en PyTorch con Adam y entropía cruzada ponderada inversamente a la frecuencia de clase. Los cuatro regímenes son: modelo local por nodo, centralizado, FedAvg y FedProx. El escalador federado se ajusta sólo con entrenamiento. La configuración por defecto es capas `(128, 64)`, 20 épocas, lote 256, aprendizaje 0.001, `weight_decay=0.0001`, 10 rondas federadas, una época local, `proximal_mu=0.01` y semilla 42. Se entrena en `train` y se evalúa preliminarmente en `validation`; `test` se reserva para evaluación final.

### 2.4 Selección del mejor modelo

La regla programada es lexicográfica: (1) minimizar coste medio de validación; (2) maximizar F1 macro en empate; (3) maximizar exactitud balanceada; y (4) minimizar parámetros. Además se exportan coste total, exactitud, F1 macro, PR-AUC macro uno-contra-el-resto, precisión, exhaustividad, F1 por clase y matriz de confusión. La exactitud no determina el ganador.

### 2.5 Validación cruzada

Se aplica validación cruzada agrupada de **5 folds**, determinista con semilla 42. La agrupación es por `vehicle_id`, con solapamiento vehículo-entrenamiento/validación igual a cero. La estratificación usa `node_id` y la máxima clase observada por vehículo. En cada fold el escalador se ajusta sólo a los folds de entrenamiento. Las asignaciones, métricas por fold e historiales se guardan como CSV.

### 2.6 Hiperparámetros y optimización

No se usa `GridSearchCV` ni Optuna: hay una cuadrícula manual y exhaustiva de cuatro MLP, evaluados en cinco folds (20 ajustes).

| Candidato | Capas ocultas | Épocas | Tasa de aprendizaje | Lote | `weight_decay` |
|---|---:|---:|---:|---:|---:|
| MLP-01 | `(64)` | 10 | 0.0010 | 256 | 0.0001 |
| MLP-02 | `(128, 64)` | 10 | 0.0010 | 256 | 0.0001 |
| MLP-03 | `(64)` | 10 | 0.0005 | 256 | 0.0001 |
| MLP-04 | `(128, 64)` | 10 | 0.0005 | 256 | 0.0001 |

Rondas federadas, épocas locales y `proximal_mu` permanecen en 10, 1 y 0.01. La configuración ganadora se serializa en `selection.json` y se reutiliza para el estudio de semillas.

### 2.7 Pruebas estadísticas robustas inferenciales

Los regímenes y políticas se comparan bajo semillas y condiciones pareadas. Para cada comparación se calcula diferencia orientada a mejora, media e intervalo percentil bootstrap bilateral al 95 % con **10 000 remuestras** y semilla determinista. La significancia se prueba mediante **Wilcoxon de rangos con signo**, bilateral: método exacto con hasta 25 diferencias no nulas sin empates y aproximación normal con corrección de continuidad y de empates en los demás casos. Se reportan estadístico, *p*, biserial de rangos y número de pares no nulos. Holm--Bonferroni corrige las comparaciones múltiples (`p_value_holm`). El estudio de taller añade no inferioridad productiva mediante el límite inferior del IC bootstrap frente a un margen predefinido.

### 2.8 Reportes Streamlit

Las vistas SCANIA son *Dashboard*, *Datos y EDA*, *Telemetría*, *Riesgo predictivo*, *Aprendizaje* y *Federación*. El selector fija *split*, nodo, vehículo y régimen. Las pestañas de aprendizaje organizan EDA, entrenamiento, selección, validación cruzada, interpretabilidad, estadística y soberanía. Las salidas incluyen tarjetas, gráficos Plotly, curvas de pérdida, tablas, métricas, importancia por permutación y reportes CSV/JSON/HTML; cada evidencia se acompaña de fuente, *split*, muestra y naturaleza (observada, derivada, mixta o simulada). La importancia por permutación expresa dependencia predictiva, no causalidad.

## 3. Resultados

### Resultados cuantitativos consolidados de la simulación

La base documental del repositorio reporta una evaluación independiente de **810 trayectorias** simuladas (9 camiones × 3 escenarios × 3 políticas × 10 semillas), **540 diferencias pareadas**, 9 casos de interoperabilidad y 10 de continuidad. Estos resultados pertenecen a datos sintéticos, no a SCANIA Component X.

| Escenario | Política | Resultado principal |
|---|---|---|
| Normal | Todas | Disponibilidad 100 %, sin fallas ni falsas alertas. |
| Degradación lineal | Sin mantenimiento | Una falla por trayectoria; disponibilidad 80.74 %. |
| Degradación lineal | Umbral | Sin fallas; disponibilidad 86.67 %; tres intervenciones por trayectoria. |
| Degradación lineal | Predictiva | Sin fallas; disponibilidad aproximada 75.5--76.8 % por exceso de intervenciones. |
| Degradación variable | Sin mantenimiento | Una falla por trayectoria; disponibilidad 72.78 %. |
| Degradación variable | Umbral | Sin fallas; disponibilidad 86.67 %. |
| Degradación variable | Predictiva | Sin fallas; disponibilidad aproximada 64.4--64.6 %. |

La política de umbral ofrece, bajo esta parametrización, el mejor compromiso entre disponibilidad, fallas e intervenciones. La predictiva evita fallas pero reduce disponibilidad por sobreintervención; ello identifica necesidad de calibración, no inferioridad general del mantenimiento predictivo. La evaluación Langflow consolidada registró 7 respuestas para 7 casos consultables, bloqueo correcto del caso desconectado, latencia media de 36.84 s y máxima de 56.49 s; la fidelidad semántica exige revisión humana en `review.csv` y `review_summary.json`.

### Resultados de aprendizaje con SCANIA por generar en cada réplica

No hay artefactos `results/` versionados en esta copia; por rigor no se declaran valores de Accuracy, F1 macro, PR-AUC, coste o *p* para SCANIA sin ejecutar el protocolo. Tras una réplica, dichos valores deben extraerse de `results/model_selection/candidates.csv`, `results/model_selection/fold_metrics.csv`, `results/final_learning/summary.json` y `results/final_learning/statistics.csv`, reportando media, dispersión o IC 95 %, semillas, *p* Holm y tamaño de efecto. No debe confundirse validación con prueba final.

> **Nota obligatoria para figuras, gráficos y tablas.** Toda figura, gráfico y tabla mostrada en pantalla o incorporada en los reportes debe llevar al pie la leyenda: *"Interpretabilidad y explicabilidad"*.

## Limitaciones

- SCANIA contiene variables anónimas del componente X; no identifica motores, frenos, neumáticos, toneladas ni operaciones mineras reales.
- La federación es local y experimental; los nodos no equivalen a organizaciones reales.
- La simulación utiliza ecuaciones y fallas sintéticas no calibradas industrialmente.
- η² e importancia por permutación son medidas descriptivas/predictivas, no pruebas causales.
