# Resumen general de MineTwin

## Propósito

MineTwin es un prototipo académico de mantenimiento predictivo basado exclusivamente en el dataset público **SCANIA Component X**. Su objetivo es estudiar cómo preparar datos históricos de camiones pesados, entrenar modelos de riesgo, comparar aprendizaje local, centralizado y federado, y presentar evidencia trazable para cada resultado.

El sistema no representa una operación minera real ni utiliza telemetría de una mina. El nombre del proyecto y su enfoque conceptual se relacionan con mantenimiento predictivo de flotas, pero la evidencia experimental procede de SCANIA Component X. Las particiones `alpha`, `beta` y `gamma` son divisiones lógicas de un único dataset público; no son contratistas u organizaciones independientes.

## Qué contiene

| Área | Contenido principal |
|---|---|
| Datos | Carga, validación de esquema, manifiesto con hashes, separación oficial de `train`, `validation` y `test`, y lectura de telemetría histórica SCANIA. |
| Preparación | Ventanas temporales causales, imputación basada solo en observaciones pasadas, indicadores de ausencia y particiones lógicas reproducibles. |
| Aprendizaje | Modelos MLP en PyTorch para cinco clases de riesgo, entrenamiento local, centralizado, FedAvg y FedProx. |
| Selección | Validación cruzada agrupada por vehículo, selección de hiperparámetros y prevención de solapamiento entre vehículos de entrenamiento y validación. |
| Evaluación | Coste de clasificación SCANIA, F1 macro, exactitud balanceada, PR-AUC macro, matrices de confusión, baselines y comparaciones estadísticas pareadas. |
| Federación | Publicación de estados de riesgo por partición sin transferir las variables crudas al coordinador experimental. |
| Interpretabilidad | Importancia por permutación de variables y resúmenes temporales; describe asociación predictiva, no causalidad. |
| Interfaz | Aplicación Streamlit con Dashboard, Datos y EDA, Telemetría, Riesgo predictivo, Aprendizaje, Federación y Asistente. |
| Langflow | Capa opcional para explicar un estado publicado con evidencia acotada; no altera predicciones ni ejecuta acciones. |
| Reporte | Generación de un reporte HTML y JSON con hashes de las fuentes usadas, resultados de aprendizaje, federación, interpretabilidad y revisión humana de Langflow. |

## Flujo funcional

```text
CSV SCANIA
  -> validación y manifiesto
  -> reproducción histórica por vehículo
  -> ventanas causales y caché de aprendizaje
  -> selección y entrenamiento de modelos
  -> procedencia verificable por semilla
  -> publicación federada de estados de riesgo
  -> diagnóstico e interpretabilidad
  -> interfaz Streamlit, Langflow y reporte final
```

## Estructura del código

| Ruta | Responsabilidad |
|---|---|
| `minetwin/data/scania/` | Contratos, carga, manifiestos, preparación, preprocesamiento, reproducción y serialización del dataset. |
| `minetwin/learning/` | Caché, modelos, escalado, entrenamiento, selección, métricas, federación, diagnóstico, interpretabilidad y procedencia. |
| `minetwin/scania_ui.py` y `minetwin/research_ui.py` | Vistas Streamlit para datos, resultados, riesgos y trazabilidad. |
| `minetwin/langflow_client.py` y `minetwin/langflow_evaluation.py` | Contexto limitado, cliente Langflow, evaluación, registro de respuestas y revisión humana. |
| `minetwin/study.py` | Estudio con múltiples semillas, pruebas estadísticas y reporte final SCANIA. |
| `tests/` | Pruebas de datos, aprendizaje, federación, Langflow, estadísticas, reporte e interfaz. |
| `results/` | Salidas generadas localmente: cachés, modelos, métricas, publicaciones, explicaciones y reportes. No es código fuente. |

## Datos requeridos

El proyecto espera ocho archivos CSV en `minetwin/data/scania/`:

- `train_operational_readouts.csv`
- `train_specifications.csv`
- `train_tte.csv`
- `validation_operational_readouts.csv`
- `validation_specifications.csv`
- `validation_labels.csv`
- `test_operational_readouts.csv`
- `test_specifications.csv`

## Trazabilidad y controles

Los artefactos de entrenamiento guardan hashes de la caché, escalador, modelos, configuración y versión de PyTorch. Los procesos de publicación, diagnóstico, interpretabilidad, Langflow y reporte verifican esa procedencia antes de consumir los resultados. Esto evita combinar una caché, un modelo o una configuración de ejecuciones distintas.

La evaluación principal usa `validation`, porque el dataset público no proporciona etiquetas de prueba en el flujo actual. El split `test` se conserva para inferencia sin etiquetas y no se presenta como evidencia de rendimiento supervisado.

## Límites del proyecto

- Component X es anónimo: no permite afirmar qué subsistema físico del camión está fallando.
- Los resultados no validan desempeño en camiones de acarreo de mina ni en condiciones operativas reales.
- La federación ocurre dentro de un único proceso experimental y no mide una infraestructura distribuida real.
- Las explicaciones Langflow son apoyo textual y requieren revisión humana; no sustituyen las métricas del modelo.
- La importancia por permutación y los resúmenes temporales no demuestran causalidad.
