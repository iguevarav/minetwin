# Inventario para demo de feria

El inventario solo lee artefactos existentes. No modifica `results/` ni recalcula predicciones.

## A. Casos de demostración

**EXISTE.** `results/11_mejora/07_demostracion/demo_cases.csv` es un CSV UTF-8 con 30 casos. Su manifiesto es `results/11_mejora/07_demostracion/manifest.json`.

### Campos y tipos

| Campo | Tipo |
| --- | --- |
| `fold` | int |
| `rank` | int |
| `vehicle_id` | string |
| `node_id` | string |
| `time_step` | float |
| `window_observations` | int |
| `label` | int |
| `most_likely_class` | int |
| `maximum_probability` | float |
| `published_decision` | int |
| `published_expected_cost` | float |
| `calibrated_probability_4` | float |
| `probability_0` | float |
| `probability_1` | float |
| `probability_2` | float |
| `probability_3` | float |
| `probability_4` | float |
| `expected_cost_0` | float |
| `expected_cost_1` | float |
| `expected_cost_2` | float |
| `expected_cost_3` | float |
| `expected_cost_4` | float |
| `selected` | bool |
| `stratum` | string |
| `priority` | string |

### Estratificación

**EXISTE.** El manifiesto indica hasta dos casos por nodo experimental × clase retrospectiva.

| Nodo | Clase retrospectiva | Casos |
| --- | --- | --- |
| alpha | 0 | 2 |
| alpha | 1 | 2 |
| alpha | 2 | 2 |
| alpha | 3 | 2 |
| alpha | 4 | 2 |
| beta | 0 | 2 |
| beta | 1 | 2 |
| beta | 2 | 2 |
| beta | 3 | 2 |
| beta | 4 | 2 |
| gamma | 0 | 2 |
| gamma | 1 | 2 |
| gamma | 2 | 2 |
| gamma | 3 | 2 |
| gamma | 4 | 2 |

| Dato solicitado | Estado | Ruta y campo |
| --- | --- | --- |
| Vehículo, nodo, corte, tamaño de ventana, fold y clase retrospectiva | EXISTE | `demo_cases.csv`: `vehicle_id`, `node_id`, `time_step`, `window_observations`, `fold`, `label`. |
| Modelo y semilla | PARCIAL | Globales, no por caso: `manifest.json` → `priority.model=mlp`, `priority.seed=42`. |
| Régimen de entrenamiento | NO EXISTE | La cola OOF procede de selección centralizada por fold; no guarda un régimen `local`, `centralized`, `fedavg` o `fedprox`. |
| Temperatura | PARCIAL | Por fold, no por caso: `05_politica_prioridad_v2/calibration_folds.csv` → `temperature`; puede unirse con `fold`. |
| Cinco probabilidades calibradas | EXISTE | `probability_0` a `probability_4`; `calibrated_probability_4` repite la quinta. |
| Clase más probable y decisión por costo | EXISTE | `most_likely_class`, `published_decision`. |
| Costos esperados | EXISTE | `expected_cost_0` a `expected_cost_4`, y mínimo publicado en `published_expected_cost`. |
| Puesto y selección por capacidad | EXISTE | `rank`, `selected`; la capacidad publicada es 20 en `policy.json`. |

Muestra: `docs/demo_muestras/caso_ejemplo.json`.

## B. Telemetría de los casos

| Pregunta | Estado | Resultado |
| --- | --- | --- |
| Recuperar historia cruda hasta el corte | EXISTE | Fuente: `minetwin/data/scania/train_operational_readouts.csv`. Funciones: `ScaniaReplayStore.readouts(DatasetSplit.TRAIN, vehicle_id)` para lectura indexada y `ScaniaDataset.iter_readouts(DatasetSplit.TRAIN)` para recorrido secuencial. Cada lectura trae `time_step` y las 105 variables; ausencias como `None` en `FeatureReadout`. |
| Lecturas hasta el corte | EXISTE | Mínimo 5, mediana 43.5, máximo 178 en los 30 casos. |
| Índices de inicio y fin de ventana causal | PARCIAL | No se persisten como columnas. Se derivan de historia cruda y `window_observations`: inicio = lecturas hasta corte − observaciones de ventana + 1; fin = lecturas hasta corte. Hay 28 casos con 12 observaciones, uno con 8 y uno con 5. Se incluyen para el caso de muestra. |
| Familia `397_*` | EXISTE | 36 bins en orden de cabecera: `397_0, 397_1, 397_2, 397_3, 397_4, 397_5, 397_6, 397_7, 397_8, 397_9, 397_10, 397_11, 397_12, 397_13, 397_14, 397_15, 397_16, 397_17, 397_18, 397_19, 397_20, 397_21, 397_22, 397_23, 397_24, 397_25, 397_26, 397_27, 397_28, 397_29, 397_30, 397_31, 397_32, 397_33, 397_34, 397_35`. |
| Variables `171_0`, `666_0`, `427_0` | EXISTE | Las tres aparecen en la cabecera de telemetría. |
| Variables no decrecientes por vehículo | EXISTE: NO | 118 de 23550 vehículos de train (0.5011%) presentan al menos un descenso observado en alguna de las 105 variables. Se compararon lecturas consecutivas no ausentes dentro de cada vehículo. |

Muestra: `docs/demo_muestras/historia_ejemplo.csv` contiene las últimas 20 lecturas del caso JSON, con `time_step`, `171_0`, `666_0`, `427_0` y los 36 bins `397_*`. Los ausentes se escriben como `NaN`.

## C. Cola y política

**EXISTE.** `results/11_mejora/05_politica_prioridad_v2/oof_priority_queue.csv`.

- Columnas: `fold, rank, vehicle_id, node_id, time_step, window_observations, label, most_likely_class, maximum_probability, published_decision, published_expected_cost, calibrated_probability_4, probability_0, probability_1, probability_2, probability_3, probability_4, expected_cost_0, expected_cost_1, expected_cost_2, expected_cost_3, expected_cost_4, selected`.
- Vehículos: 23550 registros OOF, uno por vehículo y por su última ventana causal retenida.
- Alcance: **PARCIAL para una cola global**. El archivo agrupa cinco folds, pero `rank` y `selected` reinician por fold. Es decir, contiene cinco colas OOF, no una cola global única.
- `P(clase 4)` calibrada: **EXISTE** en `calibrated_probability_4` para los 23550 registros.
- Capacidades evaluadas: 20, 50, 100 y 250. Publicada: 20, en `policy.json` → `selected_capacity`.

| K por fold | Recall medio | Precisión media | Costo medio de cola | Publicada |
| --- | --- | --- | --- | --- |
| 20 | 0.03 | 0.6 | 3.97 | Sí |
| 50 | 0.0675 | 0.54 | 4.564 | No |
| 100 | 0.126 | 0.504 | 4.895999999999999 | No |
| 250 | 0.265 | 0.42400000000000004 | 5.6808 | No |

Muestra: `docs/demo_muestras/cola_ejemplo.csv` contiene las primeras 10 filas originales.

## D. Modelo e interpretabilidad

| Dato | Estado | Ruta y detalle |
| --- | --- | --- |
| Representación, variables y arquitectura | EXISTE | `04_comparacion_v2/selection.json`: `baseline`, 527 variables, MLP con una capa oculta de ancho 64. |
| Modelos finales y escalador | EXISTE | `06_evaluacion_final_v2/models/`: `centralized.pt`, `fedavg.pt`, `fedprox.pt`, `local_alpha.pt`, `local_beta.pt`, `local_gamma.pt`, `scaler.npz`. |
| Checkpoints OOF por fold | NO EXISTE | La política entrena para producir OOF y no serializa los cinco modelos temporales. |
| Temperaturas OOF | PARCIAL | Cinco valores en `05_politica_prioridad_v2/calibration_folds.csv`, sin objeto calibrador serializado. Recalcular exactamente un caso OOF exige reentrenar el fold. |
| Atribución por vehículo | NO EXISTE | No hay importancia local por caso. |
| Importancia global por permutación | EXISTE | `06_evaluacion_final_v2/interpretability/permutation_importance.csv`; FedProx en validation, tres permutaciones por grupo. |

| Rango | Variable/grupo | Aumento medio de costo | Desviación |
| --- | --- | --- | --- |
| 1 | 167_6 | 0.058197912537983974 | 0.0035745142709651396 |
| 2 | 397_2 | 0.039767472585546436 | 0.01420297602804088 |
| 3 | 397_32 | 0.03210463733650334 | 0.0050564212533386445 |
| 4 | 459_10 | 0.03091557669441099 | 0.006816243964296317 |
| 5 | 171_0 | 0.03045316422248649 | 0.003935360287649207 |
| 6 | 397_8 | 0.03045316422248649 | 0.0023891057365603525 |
| 7 | 459_19 | 0.029594398203197276 | 0.06036502695543142 |
| 8 | 291_3 | 0.028735632183908066 | 0.0038274292342104286 |
| 9 | window | 0.028207160787421987 | 0.0020339284341845494 |
| 10 | 158_1 | 0.027744748315497485 | 0.0011034015780480673 |

## E. Resultados

### Línea base OOF

**PARCIAL.** AP de clase 4 y costo medio no están como campos resumen; se derivaron de `02_linea_base/selected_oof_predictions.csv`, sin reentrenar. Las métricas de cola existen en `priority_baseline.json`.

- AP clase 4: 0.141601
- Costo medio: 9.157766

| K global | Recall@K | Precision@K | Recall aleatorio |
| --- | --- | --- | --- |
| 20 | 0.004 | 0.4 | 0.0008492569002123143 |
| 50 | 0.0115 | 0.46 | 0.0021231422505307855 |
| 100 | 0.0255 | 0.51 | 0.004246284501061571 |
| 250 | 0.0645 | 0.516 | 0.010615711252653927 |

### Política OOF calibrada

**PARCIAL.** AP de clase 4 y costo medio se derivaron de `05_politica_prioridad_v2/oof_priority_queue.csv`; las métricas por K existen en `policy_summary.csv`.

- AP clase 4: 0.325082
- Costo medio: 9.485945

Las métricas por capacidad están en la sección C. La referencia aleatoria por fold existe en `05_politica_prioridad_v2/random_baseline.csv`.

### Evaluación final en validation

**PARCIAL.** `06_evaluacion_final_v2/models/metrics.json` guarda costo medio, PR-AUC macro, exactitud balanceada y exactitud. No guarda AP de clase 4, recall@K, precision@K ni una cola de prioridad sobre validation.

| Régimen | Costo medio | PR-AUC macro | Exactitud balanceada | Exactitud |
| --- | --- | --- | --- | --- |
| local | 9.372968688069758 | 0.21264603007179236 | 0.2095229928180941 | 0.08660325009908838 |
| centralized | 9.443519619500595 | 0.21089516249907186 | 0.20685925608318145 | 0.061038446294094335 |
| fedavg | 9.496432818073721 | 0.21055588066005745 | 0.2060038589345053 | 0.05687673404676972 |
| fedprox | 9.486325802615934 | 0.21011839650900588 | 0.20620752492228536 | 0.05786761791518034 |

### Federación publicada

**EXISTE.** `06_evaluacion_final_v2/federation/summary.json`: 5046 estados, 5106 mensajes, 11152398 bytes totales. Estados serializados: 2964318 bytes. Referencia centralizada de variables: 10636968 bytes.

Muestra consolidada: `docs/demo_muestras/metricas.json`.

## F. Definiciones

### Clases retrospectivas

| Clase | Significado |
| --- | --- |
| 0 | Más de 48 pasos hasta reparación, o sin reparación durante el estudio. |
| 1 | Entre 24 y 48 pasos hasta reparación. |
| 2 | Entre 12 y 24 pasos hasta reparación. |
| 3 | Entre 6 y 12 pasos hasta reparación. |
| 4 | Hasta 6 pasos hasta reparación. |

### Matriz de costos SCANIA

Filas: clase real. Columnas: decisión publicada.

| Real \ Decisión | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 7 | 8 | 9 | 10 |
| 1 | 200 | 0 | 7 | 8 | 9 |
| 2 | 300 | 200 | 0 | 7 | 8 |
| 3 | 400 | 300 | 200 | 0 | 7 |
| 4 | 500 | 400 | 300 | 200 | 0 |

### Textos de la pestaña “Demostración de prioridad”

- “Ejemplos SCANIA reales de train con predicciones fuera de fold. No representan una evaluación final ni una orden de mantenimiento.”
- “Revisión humana prioritaria: el caso entra en la capacidad fijada por la política. No confirma una falla ni activa mantenimiento.”
- “No priorizado en la capacidad fijada. La evaluación humana sigue siendo necesaria para interpretar cualquier señal de riesgo.”

## Muestras entregadas

| Archivo | Contenido |
| --- | --- |
| `docs/demo_muestras/caso_ejemplo.json` | Caso completo con temperatura e índices de ventana derivados. |
| `docs/demo_muestras/historia_ejemplo.csv` | Últimas 20 lecturas crudas solicitadas. |
| `docs/demo_muestras/cola_ejemplo.csv` | Diez primeras filas de cola OOF. |
| `docs/demo_muestras/metricas.json` | Métricas, política, federación e importancia global recortadas. |

El total de las muestras es menor a 200 KB.
