# Plan de mejora de priorización

## Objetivo

Mejorar la capacidad de priorizar revisiones de Component X sin convertir la mayoría de la flota en alerta. El flujo conservará datos SCANIA reales, ventanas causales, particiones por vehículo y trazabilidad de artefactos.

## Fase 1: Línea base y variables de uso reciente

- Congelar una línea base con las variables y modelo vigentes, sus predicciones fuera de fold y su cola de prioridad.
- Reportar como referencia una clasificación aleatoria: el recall esperado al revisar K vehículos es K dividido entre el tamaño de la flota.
- Confirmar las familias que representan histogramas y contadores.
- Añadir incrementos causales por ventana, totales de actividad, proporciones de incrementos y señales de reinicio o total cero.
- Mantener las variables vigentes. La pendiente ya está normalizada por duración; no duplicarla.
- Versionar el esquema de variables y regenerar la caché.

## Fase 2: Selección dentro de train

- Comparar la representación vigente y la nueva con los mismos cinco folds agrupados por vehículo.
- Evaluar MLP, regresión logística e HistGradientBoostingClassifier.
- Medir costo, AP de clase 4, exactitud balanceada, recall@q y precision@q, donde q es el porcentaje fijo de flota revisado.
- Seleccionar modelo y variables solo con resultados fuera de fold.
- Registrar prevalencia de clases, AP de clase 4 y recall@q por fold; no usar las etiquetas de `validation` para ajustar variables, calibración ni umbrales.
- Aplicar calibración ajustada solo con datos del fold de entrenamiento. Una corrección de prior solo será válida si procede de una fuente operativa independiente.

## Fase 3: Política de prioridad

- Conservar la matriz de costos para el análisis.
- Calibrar probabilidades dentro de cada fold de train.
- Seleccionar una capacidad fija de revisión K mediante CV: 20, 50, 100 o 250 vehículos.
- Publicar una cola ordenada por beneficio esperado o riesgo calibrado de clase 4, no una alerta ilimitada.
- Mantener el mismo K para la línea base y todos los candidatos. Limitar la cola no cuenta como mejora predictiva.

## Fase 4: Entrenamiento y evaluación

- Reentrenar local, centralizado, FedAvg y FedProx con la configuración ganadora.
- Generar publicación federada, diagnóstico, interpretabilidad y reporte.
- Evaluar validation una vez como comparación documentada. Para una conclusión final independiente, usar una futura partición etiquetada y no inspeccionada.

## Fase 5: Demostración

- Generar un conjunto fijo y estratificado de ejemplos SCANIA reales para la interfaz.
- Mostrar vehículo, ventana, clase más probable, prioridad, costo esperado y recomendación humana.
- Diferenciar demostración, evaluación y resultados operativos.

## Criterios de adopción

### Modelo

- Mejorar frente a la línea base en AP de clase 4 y recall@q con predicciones fuera de fold.
- Confirmar que la diferencia es consistente entre folds y reportar su intervalo de confianza.
- No empeorar el costo medio de forma no justificada.

### Política operativa

- K representa capacidad de revisión y se fija antes de comparar modelos.
- Reportar precision@q, recall@q y costo de la cola; una menor cantidad de alertas por usar K no demuestra una mejora del modelo.

### Evaluación final

- Comparar una vez en `validation` sin modificar la configuración elegida.
- Documentar el cambio de prevalencia entre train, folds y validation. Una evaluación final independiente requiere una futura partición etiquetada no inspeccionada.
