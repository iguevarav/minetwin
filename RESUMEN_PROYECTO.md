# Resumen de MineTwin

MineTwin es un prototipo académico para estudiar mantenimiento predictivo con SCANIA Component X. Consume telemetría histórica pública, genera variables temporales causales y estima cinco niveles de tiempo hasta reparación. Los datos de `train` se usan para selección agrupada por vehículo; `validation` se mantiene para la evaluación supervisada final. `test` no tiene etiquetas y no se usa para informar métricas de desempeño.

El flujo vigente contiene validación de esquema y manifiestos, reproducción histórica, caché causal, selección de cuatro configuraciones MLP, comparación de cuatro variantes de pérdida y decisión, y referencias de regresión logística e HistGradientBoostingClassifier únicamente en validación cruzada. La variante operativa es pérdida sin ponderación con mínimo costo esperado.

El estudio de diez semillas compara modelos locales, centralizados, FedAvg y FedProx. Sus artefactos guardan hashes de datos, escalador y modelos. La publicación federada comparte estados de riesgo y parámetros de modelos dentro de un único proceso experimental; no representa una infraestructura distribuida de organizaciones independientes.

La interfaz Streamlit presenta siete vistas: Dashboard, Datos y EDA, Telemetría, Riesgo predictivo, Aprendizaje, Federación y Asistente. Langflow queda como capa opcional de explicación sobre estados publicados y no interviene en el resultado predictivo.

Los resultados se organizan en `results/01_datos` a `results/10_reporte`. `results/10_reporte/articulo` reúne las tablas CSV, figuras PNG a 300 dpi, procedencia y el documento `resultados.md` para el artículo. Los resultados proceden de SCANIA Component X; por ello no permiten afirmar desempeño sobre camiones de acarreo de una mina ni identificar el componente físico anónimo.
