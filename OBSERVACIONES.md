# Observaciones y trabajo pendiente de MineTwin

Este archivo registra el estado verificable del proyecto y lo que falta para sostener sus conclusiones de investigación. Sustituye el plan de transición `README_DATOS_REALES.md`.

## Estado comprobado

- SCANIA Component X es la fuente predeterminada de la interfaz. Hay reproducción temporal de lecturas, EDA, estimación de riesgo, aprendizaje, interpretabilidad y publicación federada. La simulación tiene un modo separado para estudiar operaciones que SCANIA no registra.
- La selección de arquitectura usó cinco folds agrupados por vehículo. `MLP-01` obtuvo un costo medio de `8.76376159821523` en esos folds. La comparación final usó diez semillas y el conjunto oficial de validación.
- El menor costo medio final fue el del entrenamiento local: `9.656302021403093`; FedProx obtuvo `9.744252873563218` y FedAvg `9.771006738010305`. Ningún método federado demostró superioridad robusta frente al local después de la corrección de Holm. Los detalles están en `results/final_learning/`.
- La publicación FedProx de validación generó 5 046 estados; el coordinador registró cero variables y cero registros crudos. Esto acredita el contrato de intercambio del prototipo, no un despliegue entre organizaciones independientes.
- La importancia por permutación ya está calculada en `results/final_interpretability/` y se muestra en la vista de aprendizaje.

## Pendientes prioritarios

| Prioridad | Trabajo | Evidencia o criterio de cierre |
|---|---|---|
| Alta | Evaluar y mejorar la utilidad predictiva. Comparar con reglas constantes y otros modelos sencillos; revisar desbalance, calibración de probabilidades, matriz de costos y rendimiento por clase y nodo. Mantener la validación oficial aislada de la selección. | El resultado actual tiene F1 macro de `0.02188446661703463` para Local y `0.011793329509019467` para FedProx. Registrar nuevas métricas y explicar tanto mejoras como resultados negativos. No presentar el modelo actual como apto para mantenimiento operativo. |
| Alta | Adaptar el asistente y sus casos de prueba a SCANIA. Hoy `minetwin/assistant_ui.py`, `minetwin/langflow_client.py` y `minetwin/langflow_evaluation.py` consumen vistas y escenarios simulados. | El contexto del flujo contiene solo vehículo, nodo, split, instante, clase, probabilidades, calidad, variables anónimas y modelo publicados para una observación concreta. La evaluación cubre las cinco clases, ausentes, desconexión, error y timeout; no atribuye componentes físicos ni ejecuta acciones. |
| Alta | Completar el estudio factorial del taller simulado. `results/final_workshop/` y `results/final_workshop_2/` tienen manifiesto `incomplete`; `results/phase4_workshop/` es una comparación preliminar. | Generar un artefacto con manifiesto `complete`, diez semillas pareadas, tres escenarios, tres capacidades de taller, tres razones de costo, pruebas estadísticas y no inferioridad. Solo entonces generar el reporte final y exponer esas conclusiones como resultados del simulador. |
| Alta | Resolver el alcance del título de investigación. SCANIA corresponde a vehículos pesados y no identifica camiones de acarreo, minas a cielo abierto ni contratistas. | Incorporar un dataset verificable de acarreo minero para validar esa población, o declarar explícitamente que SCANIA es una prueba de transferencia metodológica y que el caso minero se estudia mediante simulación. |
| Media | Verificar la soberanía entre nodos independientes. Los nodos actuales funcionan como particiones lógicas dentro de un proceso. | Ejecutar una demostración con procesos o servicios separados y comprobar que el coordinador recibe únicamente parámetros y estados permitidos; si no se implementa, limitar la afirmación a federación lógica experimental. |
| Media | Cerrar la revisión del asistente. La última evaluación guardada recibió siete respuestas y bloqueó el caso desconectado, pero usa escenarios simulados y `review.csv` no contiene puntuaciones humanas. | Exportar la definición del flujo; registrar versión de flujo, prompt, modelo y observación; completar la revisión factual de los nuevos casos SCANIA y producir `review_summary.json`. |
| Media | Hacer reproducible la demostración desde un clon limpio. `results/` y los CSV originales están excluidos de Git, pero varias vistas leen artefactos de `results/`. | Documentar la obtención del dataset y la secuencia de comandos para regenerar los artefactos mínimos, con sus hashes y versiones. Actualizar `README.md`, que aún describe principalmente la simulación. |
| Media | Ejecutar la verificación final del estado actual. | Suite completa, Ruff y recorrido manual de las vistas SCANIA y simulación sin errores; registrar fecha y resultado de cada ejecución. |

## Comandos de verificación y generación pendientes

Cada comando que produce artefactos requiere un directorio de salida nuevo. Desde la raíz del proyecto, en PowerShell:

```powershell
python -m minetwin.study workshop --output results\final_workshop_nuevo
python -m minetwin.study report --learning results\final_learning --workshop results\final_workshop_nuevo --scania results\phase1_scania --federation results\final_federation --selection results\final_selection --output results\final_report_nuevo
python -m pytest -q -p no:cacheprovider
python -m ruff check minetwin tests
python -m streamlit run app.py
```

El reporte requiere que el estudio del taller termine con manifiesto `complete`. La evaluación Langflow sobre SCANIA requiere primero el cambio de código indicado arriba y una instancia de Langflow configurada; la evaluación simulada existente no valida explicaciones sobre SCANIA.

## Límite de las conclusiones actuales

Los resultados reales permiten estudiar clasificación de riesgo de Component X, heterogeneidad entre nodos y costos de intercambio en una federación lógica. El experimento simulado permite estudiar taller, producción y políticas de mantenimiento. Por ahora no hay evidencia de mejora predictiva federada sobre el modelo local, ni de eficacia operativa en una flota minera real.
