# MineTwin: transición a una aplicación basada en datos reales

## Objetivo

MineTwin debe utilizar SCANIA Component X como fuente principal para el análisis exploratorio, entrenamiento, validación, inferencia y federación. La simulación se conservará únicamente para estudiar políticas operativas que el dataset real no permite observar.

La aplicación actual es híbrida. La sección de aprendizaje consume SCANIA, mientras que Resumen, Telemetría, Alertas, Mantenimiento, Coordinación y los casos originales de Langflow todavía utilizan camiones simulados.

## Límite de los datos disponibles

SCANIA Component X proporciona lecturas operativas anónimas, especificaciones, tiempos hasta reparación y clases de riesgo. No contiene:

- Ubicación o recorrido en una mina.
- Ciclos y toneladas de acarreo.
- Bahías o capacidad de taller.
- Órdenes e intervenciones ejecutadas.
- Costos reales de mantenimiento.
- Identidad de contratistas mineros.

Por esta razón, las vistas basadas en SCANIA no deben mostrar mapas, motores, frenos, neumáticos, toneladas ni reparaciones como observaciones reales. La optimización del taller seguirá siendo un experimento simulado y deberá identificarse explícitamente.

## Navegación objetivo

| Vista | Fuente | Contenido |
|---|---|---|
| Dashboard | SCANIA | Vehículos, nodos, clases de riesgo y calidad |
| Datos y EDA | SCANIA | Distribuciones, ausentes, desbalance y variables |
| Aprendizaje | SCANIA | Entrenamiento, convergencia e hiperparámetros |
| Validación | SCANIA | Validación cruzada, métricas, selección y estadística |
| Federación | SCANIA | Estados publicados, datos retenidos y transferencias |
| Asistente | SCANIA y Langflow | Explicación del riesgo de Component X |
| Experimento operativo | Simulación | Taller, producción y políticas P0 a P3 |

## Fase 1: separar datos reales y simulación

### Resultado

SCANIA será la fuente predeterminada de la aplicación y la simulación quedará aislada como experimento operativo.

### Implementación

- Crear una sesión de reproducción por vehículo a partir de sus readouts reales.
- Seleccionar split, nodo y vehículo desde la interfaz.
- Evitar que objetos simulados entren en vistas SCANIA.
- Etiquetar cada resultado como observado, derivado o simulado.
- Mover los controles de escenario, avance y reparación a Experimento operativo.

### Criterio de cierre

Ninguna vista de datos reales presenta temperatura de motor, neumáticos, mapa, toneladas o reparaciones inventadas.

## Fase 2: rehacer las vistas principales

### Dashboard

- Cantidad de vehículos y readouts.
- Distribución por nodo.
- Distribución de clases de riesgo.
- Calidad y proporción de valores ausentes.
- Modelo y split activos.

### Datos y EDA

- Distribución de clases.
- Distribución por nodo.
- Valores ausentes por conjunto.
- Variables derivadas con mayor separación entre clases.
- Significado de las clases temporales.
- Tamaño de muestra y fuente junto a cada resultado.

### Telemetría

- Variables anónimas de Component X.
- Secuencia temporal real del vehículo seleccionado.
- Ventana causal empleada por el modelo.
- Calidad, ausentes e imputaciones causales.
- Ausencia de nombres físicos que el dataset no proporciona.

### Alertas y mantenimiento

- Clase predicha y probabilidades.
- Nivel de riesgo y recomendación.
- Evidencia que sustenta la recomendación.
- Estado no estimable cuando faltan datos suficientes.
- Sin ejecución de reparaciones en el modo SCANIA.

### Federación

- Partición de vehículos por nodo.
- Registros retenidos localmente.
- Estados de riesgo publicados.
- Parámetros intercambiados por ronda.
- Bytes enviados y reducción frente a centralización.
- Registros crudos recibidos por el coordinador, que debe permanecer en cero.

### Criterio de cierre

Toda cifra indica su fuente, split, tamaño de muestra y naturaleza observada, derivada o simulada.

## Fase 3: interpretación del aprendizaje

Cada estadística, gráfico o tabla deberá responder:

1. Qué mide.
2. Qué dirección representa una mejora.
3. Qué resultado se obtuvo.
4. Qué conclusión permite.
5. Qué limitación conserva.

### EDA

- Explicar el desbalance de clases y por qué la exactitud simple es insuficiente.
- Interpretar la heterogeneidad entre nodos.
- Diferenciar valores ausentes de ceros observados.
- Presentar eta cuadrado como asociación descriptiva y no causalidad.

### Entrenamiento

- Mostrar pérdida por época o ronda.
- Identificar convergencia, inestabilidad o ausencia de mejora.
- Separar pérdida de entrenamiento y rendimiento de validación.
- Indicar que una pérdida menor no garantiza menor costo operativo.

### Hiperparámetros

- Mostrar arquitectura, épocas, lote, tasa de aprendizaje y regularización.
- Comparar costo medio y variación entre folds.
- Explicar la regla exacta de selección.
- Registrar el candidato elegido y su configuración completa.

### Selección del modelo

- Seleccionar primero la arquitectura mediante validación cruzada sobre entrenamiento.
- Aplicar la arquitectura elegida a Local, Centralizado, FedAvg y FedProx.
- Comparar los regímenes sobre la validación oficial.
- Priorizar el costo SCANIA y usar F1 macro y exactitud balanceada como métricas complementarias.

### Interpretabilidad del modelo

- Calcular importancia por permutación usando aumento del costo medio.
- Agrupar características por variable de origen y estadístico temporal.
- Mostrar probabilidades por clase para el vehículo seleccionado.
- Explicar que las variables de SCANIA son anónimas y no permiten atribución física directa.

### Criterio de cierre

Una persona puede explicar cada resultado sin deducir manualmente el significado de una tabla extensa.

## Fase 4: validación y estadística explicables

### Validación cruzada

- Cinco folds agrupados por `vehicle_id`.
- Ningún vehículo puede aparecer simultáneamente en ajuste y validación.
- Estratificación por nodo y máxima clase de riesgo observada.
- Escalado calculado únicamente con el subconjunto de entrenamiento de cada fold.
- Costo, F1 macro, exactitud balanceada y PR AUC por fold.

### Evaluación final

- Diez semillas pareadas.
- Regímenes Local, Centralizado, FedAvg y FedProx.
- Mismo conjunto oficial de validación para todos los regímenes.
- Resultados globales, por nodo y por clase.

### Pruebas robustas

- Wilcoxon pareado.
- Corrección de Holm por comparaciones múltiples.
- Intervalos bootstrap.
- Correlación biserial de rangos como tamaño del efecto.
- Prueba de no inferioridad para producción en el experimento operativo.

### Regla de interpretación

Una mejora se considera respaldada cuando:

- La mejora media favorece al candidato.
- El intervalo de confianza no cruza cero.
- El valor p ajustado es menor que 0.05.
- El tamaño del efecto tiene magnitud relevante.

Si estas condiciones no se cumplen, la aplicación debe indicar que no existe evidencia suficiente para afirmar superioridad.

### Criterio de cierre

Cada comparación comunica resultado observado, incertidumbre, significancia, tamaño del efecto y limitación.

## Fase 5: Langflow sobre datos reales

### Contexto permitido

- Identificador de vehículo y nodo.
- Split y marca temporal del readout.
- Clase predicha y probabilidades.
- Calidad y proporción de ausentes.
- Características derivadas relevantes.
- Modelo, flujo y versión del prompt.

### Restricciones

- Utilizar únicamente el nombre Component X.
- No inventar motores, frenos, neumáticos, producción o fallas mineras.
- No crear órdenes ni modificar predicciones.
- Bloquear consultas con estados desconectados o desactualizados.
- Vincular cada explicación con una observación concreta.

### Evaluación

- Casos representativos de las cinco clases.
- Datos incompletos.
- Nodo desconectado.
- Respuesta vacía, error y timeout.
- Exactitud factual, identificación de riesgo, incertidumbre y ausencia de acciones no sustentadas.

### Criterio de cierre

Las explicaciones se basan exclusivamente en evidencia SCANIA publicada por el gemelo.

## Fase 6: experimento operativo simulado

La coordinación del taller continuará como evidencia separada para estudiar decisiones que SCANIA no contiene.

### Contenido

- Políticas P0, P1, P2 y P3.
- Tres escenarios operativos.
- Diez semillas pareadas.
- Sensibilidad con una, dos y tres bahías.
- Razones de costo correctivo a preventivo de 3, 5 y 10.
- Producción, disponibilidad, fallas, intervenciones, cola y costo.

### Presentación

- Etiqueta visible de experimento simulado.
- Sin mezclar resultados con la validación SCANIA.
- Explicación de los supuestos de producción y mantenimiento.
- Conclusiones restringidas al simulador.

### Criterio de cierre

El usuario puede distinguir inmediatamente evidencia real, resultado derivado y experimento sintético.

## Fase 7: limpieza y verificación

### Limpieza

- Eliminar controles simulados de las vistas SCANIA.
- Retirar mapas y nombres físicos sin respaldo del dataset.
- Presentar interpretaciones antes de las tablas completas.
- Mantener los detalles numéricos en expansores.
- Actualizar el reporte HTML con la misma estructura de la interfaz.

### Pruebas

- Separación oficial de train, validation y test.
- Ausencia de fuga entre vehículos en validación cruzada.
- Reproducción temporal determinista de readouts.
- Imputación causal sin uso de observaciones futuras.
- Equivalencia JSON, XML y CSV.
- Datos crudos ausentes en el coordinador.
- Continuidad local durante desconexión.
- Explicaciones de Langflow sin afirmaciones físicas inventadas.
- Navegación SCANIA sin avance de simulación.
- Suite completa de pytest, Ruff y revisión visual.

### Criterio de cierre

La aplicación funciona con SCANIA como fuente principal, separa el experimento simulado y limita sus conclusiones a la evidencia disponible.

## Alcance científico resultante

SCANIA respalda:

- Análisis exploratorio con datos reales.
- Entrenamiento y validación de modelos.
- Comparación Local, Centralizada, FedAvg y FedProx.
- Federación lógica y soberanía de datos.
- Recomendaciones predictivas sobre Component X.
- Explicaciones generadas a partir de evidencia real.

La simulación respalda:

- Decisiones de mantenimiento.
- Capacidad limitada del taller.
- Producción y disponibilidad.
- Comparación operativa de P0 a P3.

La partición por nodos representa una federación experimental. No demuestra despliegue distribuido entre contratistas reales. El dataset corresponde a vehículos pesados SCANIA y no contiene operaciones reales de camiones de acarreo en una mina a cielo abierto.
