# Plan de transformación de MineTwin a una aplicación basada en datos reales

## Decisión de alcance

MineTwin conservará el título *A Federated Digital Twin Framework for Predictive Maintenance of Heterogeneous Haul Truck Fleets in Open-Pit Operations* como objetivo de diseño. La evaluación cuantitativa disponible se realizará con SCANIA Component X, un conjunto público de lecturas y reparaciones reales de camiones pesados. El trabajo describirá este uso como prueba de concepto industrial, sin afirmar que mide eficacia en una mina o soberanía entre contratistas reales.

La aplicación y el reporte final mostrarán únicamente hechos observados en SCANIA o resultados calculados a partir de sus registros. Las particiones federadas se identificarán como experimentales. La simulación de operaciones se conservará, si se necesita, fuera de la aplicación y del reporte principal.

## Punto de partida

Ya existen preparación y reproducción temporal de SCANIA, EDA, predicción de riesgo, selección por cinco folds agrupados por vehículo, evaluación de cuatro regímenes con diez semillas, importancia por permutación e intercambio federado sin registros crudos en el coordinador. La interfaz aún ofrece un modo simulado; Langflow usa ese modo; y el generador de reporte exige un estudio de taller simulado. Esos son los cambios de integración que resuelve el plan.

## Fase 1. Una sola aplicación con evidencia SCANIA

**Código:** dejar `app.py` y `minetwin/ui.py` con navegación SCANIA como única entrada de la aplicación final. Reutilizar `minetwin/scania_ui.py` y `minetwin/research_ui.py`; retirar de esa navegación el mapa, toneladas, fallas sintéticas, órdenes ejecutadas, bahías y políticas P0–P3. Mantener el código simulador aislado para no mezclar sus resultados con los reales. Unificar en un solo lugar la resolución de rutas a dataset y artefactos `results/`.

**Vistas finales:** Dashboard; Datos y EDA; Telemetría histórica; Riesgo de Component X; Aprendizaje y validación; Federación; Asistente. La vista de riesgo mostrará clase, probabilidades, calidad, observación utilizada y recomendación de inspección. Los registros de `train_tte.csv` podrán presentarse como tiempo hasta la primera reparación observada, nunca como una orden creada por MineTwin.

**Cierre:** todas las pantallas identifican fuente, split, vehículo o tamaño de muestra y si el valor es observado, derivado o partición experimental. La aplicación no presenta cifras operativas fabricadas ni llama «tiempo real» a la reproducción histórica.

## Fase 2. Evaluación predictiva que se pueda defender

**Código:** reutilizar la preparación causal y los folds actuales. Añadir baselines sencillos bajo la misma matriz de costo; revisar el desbalance entre entrenamiento y validación, calibración de probabilidades y comportamiento por clase y nodo. Comparar mejoras de arquitectura y decisión sin usar la validación oficial para elegir repetidamente hiperparámetros. Conservar Local, Centralizado, FedAvg y FedProx con semillas pareadas y el mismo conjunto de evaluación. Mostrar matriz de confusión, costo, F1 macro, exactitud balanceada y PR AUC con una interpretación breve y sus límites.

**Cierre:** un artefacto reproducible identifica datos, transformaciones, modelo, configuración, semillas y métricas. La interfaz señala si el modelo no alcanza una calidad práctica; una mejora federada solo se afirma si las pruebas pareadas y la corrección por comparaciones múltiples la respaldan. El conjunto de prueba sin etiquetas disponibles no genera métricas inventadas.

## Fase 3. Federación experimental sobre registros reales

**Código:** mantener las lecturas y ventanas SCANIA dentro de cada nodo experimental; publicar únicamente probabilidades, clase, calidad y metadatos permitidos. Reutilizar el ledger de transferencias y las métricas de comunicación existentes. Verificar que los modelos locales y los estados publicados se vinculan a la misma versión de datos y configuración. Evitar nombres de contratistas donde solo existe una partición derivada de especificaciones.

**Cierre:** el coordinador registra cero lecturas, variables y filas crudas; el reporte distingue bytes de estados publicados y bytes de parámetros del entrenamiento. Los resultados indican explícitamente que los nodos son particiones de una fuente real común y que no se ha medido un despliegue entre contratistas.

## Fase 4. Langflow sobre una observación SCANIA

**Código:** cambiar `minetwin/langflow_client.py`, `minetwin/assistant_ui.py` y `minetwin/langflow_evaluation.py` para recibir el estado publicado del vehículo seleccionado. El contexto incluirá identificador, nodo, split, paso temporal, clase, probabilidades, calidad, variables anónimas relevantes, modelo y versión de prompt. El asistente solo explicará la evidencia y no cambiará predicciones ni creará órdenes. Bloqueará estados ausentes, desactualizados o desconectados.

**Evaluación:** construir casos con datos reales que cubran clases disponibles, ausentes y errores de conexión. Registrar respuesta, latencia, límite de palabras, inmutabilidad y revisión humana de exactitud; guardar la definición y versión del flujo. El flujo anterior con casos de motor, frenos y neumáticos simulados no será evidencia para esta fase.

**Cierre:** la interfaz muestra explicaciones ligadas a una observación SCANIA concreta, sin atribuir una pieza física desconocida ni una acción ejecutada.

## Fase 5. Reporte, reproducibilidad y cierre

**Código:** adaptar `minetwin/study.py` para que el reporte final consuma solo dataset, selección, aprendizaje, estadísticas, interpretabilidad, federación y evaluación Langflow basada en SCANIA. No exigir `final_workshop` ni incluir sus métricas. Actualizar `README.md` con alcance, instalación, comandos para regenerar artefactos, procedencia, resultados exactos y límites de generalización. Conservar `OBSERVACIONES.md` como lista de pendientes y registrar allí el cierre de cada fase.

**Ejecuciones externas:** descargar los ocho CSV oficiales de SCANIA, configurar Langflow y Ollama si se usa el asistente, ejecutar la generación de artefactos en directorios nuevos, correr `python -m pytest -q -p no:cacheprovider`, `python -m ruff check minetwin tests` y revisar manualmente todas las vistas. Estas ejecuciones se indicarán con comandos; no forman parte de los cambios de código de cada fase.

**Cierre:** un clon limpio puede reconstruir la evaluación con el dataset público y los comandos documentados. El reporte no mezcla resultados simulados con SCANIA. Las conclusiones se limitan a la prueba de concepto con camiones pesados y a la federación experimental; la validación en mina queda declarada como trabajo futuro.

## Orden de uso de los documentos

Implementar este plan por fases. Al terminar cada fase, actualizar `OBSERVACIONES.md` con lo resuelto y lo que siga pendiente. Actualizar el `README.md` al consolidar el comportamiento y los resultados finales, para que describa el sistema que efectivamente se puede ejecutar.
