# Estado actual de MineTwin

## Propósito vigente

MineTwin analiza mantenimiento predictivo con datos públicos reales de SCANIA Component X. Construye ventanas causales de telemetría, compara aprendizaje local, centralizado, FedAvg y FedProx, y prioriza una capacidad finita de revisión humana. Las particiones `alpha`, `beta` y `gamma` son nodos experimentales de un mismo dataset; no representan contratistas ni una operación minera real.

## Ajuste principal

El flujo anterior podía convertir la decisión de mínimo costo esperado en alertas amplias. El flujo vigente separa tres conceptos:

- Probabilidad de cada clase.
- Decisión que minimiza el costo esperado de la matriz SCANIA.
- Prioridad operativa: una cola limitada de vehículos ordenada por la probabilidad calibrada de clase 4.

La capacidad de revisión no se presenta como mejora predictiva. La mejora se evalúa con AP de clase 4, recall y precisión a una fracción fija de la flota, además del costo medio.

## Fase 1: línea base y variables

- Se congeló una representación base de 527 variables y una línea base de prioridad con predicciones fuera de fold.
- Se añadieron variables causales opcionales de uso reciente: incrementos por ventana, proporciones por familia de histogramas, actividad total, incrementos nulos y descensos.
- Las familias de histogramas se identifican desde los nombres de variables; no se duplicó la pendiente temporal existente.
- La caché registra configuración y versión del esquema de variables.

Archivos principales: `minetwin/learning/data.py`, `minetwin/learning/contracts.py`, `minetwin/learning/priority.py` y `minetwin/learning/selection.py`.

## Fase 2: selección dentro de train

- La comparación usa cinco folds agrupados por vehículo y solo la última ventana causal de cada vehículo retenido.
- Compara representación base y de uso con MLP, regresión logística e HistGradientBoostingClassifier.
- Registra prevalencia por fold, AP de clase 4, recall y precisión a la fracción de revisión, costo medio y exactitud balanceada.
- `validation` no interviene en la selección de variables, modelos ni métricas de prioridad.

La ejecución disponible en `results/11_mejora/04_comparacion_v2` seleccionó la representación base y el MLP. La representación base conserva 527 variables; la de uso incorpora 747.

Archivos principales: `minetwin/learning/comparison.py` y `minetwin/learning/candidates.py`.

## Fase 3: política de prioridad

- La calibración por temperatura se ajusta solamente con vehículos del tramo de entrenamiento de cada fold externo.
- La prioridad se ordena por probabilidad calibrada de clase 4.
- Se comparan capacidades de 20, 50, 100 y 250 revisiones, con la misma capacidad para todos los candidatos y la referencia aleatoria.
- La política publica una cola OOF con vehículo, nodo, ventana, probabilidades, costos esperados, clase más probable, decisión de costo y selección por capacidad.

Archivos principales: `minetwin/learning/calibration.py` y `minetwin/learning/priority_policy.py`.

## Fase 4: entrenamiento y evaluación final

- El entrenamiento vigente usa pérdida sin ponderación y decisión por mínimo costo esperado.
- La configuración seleccionada se aplica a los regímenes local, centralizado, FedAvg y FedProx.
- La evaluación final descriptiva usa `validation` una vez, después de la selección interna.
- Se generan publicación federada, diagnóstico, interpretabilidad por permutación y un reporte con hashes de procedencia.

Este paso conserva el rol de `validation` como evaluación final descriptiva; no debe emplearse para escoger variables, calibración ni capacidad.

Archivo principal: `minetwin/learning/final_evaluation.py`.

## Fase 5: demostración

- Se generan ejemplos SCANIA reales y estratificados por nodo experimental y clase retrospectiva.
- Cada caso procede de una predicción fuera de fold sobre `train`, por lo que la demostración no se confunde con la evaluación final en `validation`.
- La interfaz incorpora la pestaña **Demostración de prioridad** dentro de **Riesgo predictivo**.
- La vista muestra vehículo, nodo, paso temporal, tamaño de ventana, modelo, semilla, probabilidades calibradas, clase más probable, decisión por costo, costo esperado y prioridad de revisión humana.
- La interfaz aclara que una prioridad no confirma una falla ni ejecuta mantenimiento.

Archivos principales: `minetwin/learning/demonstration.py` y `minetwin/scania_ui.py`.

## Flujo de resultados de mejora

| Directorio | Contenido |
| --- | --- |
| `results/11_mejora/01_cache_base` | Caché de la representación vigente. |
| `results/11_mejora/02_linea_base` | Línea base OOF de prioridad. |
| `results/11_mejora/03_cache_uso` | Caché con variables causales de uso reciente. |
| `results/11_mejora/04_comparacion_v2` | Comparación agrupada dentro de train y selección. |
| `results/11_mejora/05_politica_prioridad_v2` | Política calibrada y cola OOF ampliada. |
| `results/11_mejora/06_evaluacion_final_v2` | Entrenamiento, evaluación final, publicación, diagnóstico e interpretabilidad. |
| `results/11_mejora/07_demostracion` | Casos estratificados consumidos por la interfaz. |

Los tres últimos directorios deben generarse con los comandos de fases 3, 4 y 5 si aún no existen. La versión `v2` de la política es necesaria para que la demostración incluya probabilidades, costos esperados y ventana causal.

## Límites metodológicos

- Las probabilidades calibradas describen el comportamiento del modelo en el experimento; no son probabilidades operativas validadas en una mina.
- La clase retrospectiva se muestra solo como evidencia del dataset, nunca como una confirmación prospectiva para un operador.
- Una conclusión independiente requiere una futura partición etiquetada que no se use durante desarrollo.
- Langflow continúa siendo opcional y explicativo; no cambia predicciones, prioridad ni acciones.

## Verificación pendiente

```powershell
python -m ruff check minetwin tests
python -m pytest tests\test_learning.py tests\test_ui.py -q -p no:cacheprovider
python -m streamlit run app.py
```
