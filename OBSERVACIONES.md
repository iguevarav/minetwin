# Observaciones de MineTwin

## Estado al 1 de octubre de 2026

El código de las cinco fases de `PLAN_TRANSFORMACION_DATOS_REALES.md` está implementado. La aplicación principal utiliza SCANIA Component X y mantiene la simulación en una entrada separada. El asistente Langflow recibe un estado SCANIA publicado y el generador del reporte principal excluye el taller simulado. Esto describe el código disponible; no equivale a haber completado la evaluación final.

Los artefactos existentes en `results/final_learning/`, `results/final_federation/` y `results/final_interpretability/` proceden de ejecuciones anteriores. Sus resultados SCANIA son reales, pero no reúnen la trazabilidad verificada que exige el reporte nuevo. Como referencia histórica, la selección por cinco folds eligió `MLP-01` con costo medio `8.76376159821523`. En diez semillas, el costo medio fue `9.656302021403093` para Local, `9.771006738010305` para FedAvg y `9.744252873563218` para FedProx. El F1 macro medio fue `0.02188446661703463` para Local y `0.011793329509019467` para FedProx. Estas cifras no demuestran utilidad operativa ni superioridad federada robusta.

## Pendiente para cerrar la evaluación

1. **Generar artefactos verificados.** Aún no existen `results/final_learning_verified/`, `results/final_federation_verified/`, `results/final_diagnostics_verified/` ni `results/final_interpretability_verified/`. Deben generarse desde la caché SCANIA y la selección ya disponibles, en directorios nuevos.
2. **Evaluar Langflow con SCANIA.** Exportar el flujo realmente utilizado, registrar el identificador exacto del modelo de lenguaje y ejecutar la evaluación nueva. `results/phase4_langflow_scania/` todavía no existe. Después, puntuar las respuestas en `review.csv` y generar `review_summary.json`. Las evaluaciones antiguas con escenarios simulados no sustituyen esta evidencia.
3. **Generar el reporte final.** `results/final_report_scania/` todavía no existe. El comando `python -m minetwin.study report` ya no exige `--workshop`; comprueba que los datos, el modelo, la publicación, la interpretabilidad y Langflow coincidan antes de producir el informe.
4. **Actualizar `README.md`.** Aún contiene comandos y resultados del flujo simulado anterior. Debe documentar cómo obtener los CSV públicos, reconstruir los artefactos desde un clon limpio, ejecutar la nueva evaluación y reportar los resultados efectivamente obtenidos. No deben escribirse cifras finales hasta generar los artefactos nuevos.
5. **Verificar el estado final.** Ejecutar `python -m ruff check minetwin tests`, `python -m pytest -q -p no:cacheprovider` y recorrer manualmente las vistas SCANIA y la entrada separada de simulación. No hay un resultado de estas comprobaciones registrado para el código actual.

## Límites de la investigación

SCANIA Component X contiene registros reales de camiones pesados, pero no identifica operaciones de acarreo en una mina. Los nodos federados son particiones experimentales de una fuente común en un mismo proceso; no demuestran soberanía de datos entre contratistas independientes. La calidad predictiva observada en los resultados anteriores es baja y debe evaluarse frente a los baselines, el desbalance y las métricas por clase y nodo antes de sugerir uso operativo.

Una validación minera futura requerirá datos de camiones de acarreo con secuencia temporal, condiciones de operación y eventos de falla o reparación. Para demostrar federación entre organizaciones también harían falta nodos y procedencia organizativa independientes. Ninguna de esas validaciones es requisito para cerrar la prueba de concepto con SCANIA, pero sí para afirmar eficacia en mina.
