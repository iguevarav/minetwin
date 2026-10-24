# Tarea: integrar la demo "Un camión en la cola del taller" (v2)

Este documento **reemplaza** cualquier instrucción anterior sobre la demo de feria (la v1 con "¿Y si…?" y "por qué" por sensor queda descartada). Léelo completo y trabaja por fases.

## 1. Qué es

Es una vista de Streamlit para público no técnico. Tiene 4 secciones: **3 camiones** (uno que requiere revisión, uno en un punto medio y uno que no requiere revisión) y **Toda la flota**.

- Cada camión recorre 5 etapas: El camión, Sus datos, Procesamiento, Resultado y Conclusión.
- Toda la flota tiene 2 etapas: ¿Funciona? y Aprender sin compartir datos (federación y límites).

**La interfaz ya está terminada** en `flujo.html`. Solo lee un JSON (el *payload*); no calcula ninguna predicción. Tu trabajo es:

1. Construir el payload **exclusivamente con artefactos existentes**.
2. Integrarlo en Streamlit.
3. Escribir tests.

**No se reentrena ni se recalcula ningún modelo.**

## 2. Archivos entregados

| Archivo | Destino | Qué es |
|---|---|---|
| `flujo.html` | `minetwin/demo/flujo.html` | Plantilla final. **No la modifiques.** |
| `ejemplo_payload.json` | `minetwin/demo/ejemplo_payload.json` | Referencia del contrato. El camión 6961 es real; los camiones 2208 y 4417, el `umbral` y la `fila` son ilustrativos. |

Cómo funciona la plantilla:

- Contiene una sola vez el marcador `/*__MINETWIN_PAYLOAD__*/null`. Python lo reemplaza por el JSON.
- Sin reemplazo, muestra el ejemplo incrustado. **La plantilla ya no muestra ninguna etiqueta de fuente**, así que el aviso de que se está viendo el ejemplo debe darlo Streamlit (ver sección 7).
- Valida el payload al cargar y, si hay errores, los lista en pantalla.
- Expone `window.__MINETWIN_ESTADO = {ok, modo, errores}`.
- Si el payload no trae `federacion`, la etapa "Aprender sin compartir datos" desaparece sola.
- `federacion.nodos` es opcional (lista de 3 nombres). Si falta, se usan Alpha, Beta y Gamma.

## 3. Fuentes (todas existen)

| Dato | Fuente |
|---|---|
| Casos de demostración | `results/11_mejora/07_demostracion/demo_cases.csv` (30 casos) y `manifest.json` |
| Cola por grupo | `results/11_mejora/05_politica_prioridad_v2/oof_priority_queue.csv` (23 550 filas, 5 colas: `rank` reinicia por `fold`) |
| Capacidad publicada | `05_politica_prioridad_v2/policy.json` → `selected_capacity` (20) |
| Resultados de flota | `05_politica_prioridad_v2/policy_summary.csv` |
| Arquitectura | `04_comparacion_v2/selection.json` (base, 527 variables, MLP de 1 capa oculta de 64) |
| Telemetría cruda | `ScaniaReplayStore.readouts(DatasetSplit.TRAIN, vehicle_id)` |
| Resumen de la ventana | `WindowVectorizer` en `minetwin/learning/data.py` |
| Federación | `06_evaluacion_final_v2/federation/summary.json` |
| Clases | La definición que ya usa la app ("Significado de las clases") |

## 4. Fase 0: verificación (antes de programar)

Confirma y reporta:

1. Los nombres y el orden de las **5 estadísticas por variable** que produce `WindowVectorizer`, y que `105 × 5 + 2 = 527` (los 2 extra corresponden al grupo `window`).
2. Que puedes obtener esas 5 estadísticas **sin escalar** para 171_0, 666_0 y 427_0 en la ventana de un caso.
3. El número de filas por `fold` en la cola (≈ 4 710).

**Detente y espera confirmación.**

## 5. Selección de los 3 camiones

Usa una regla fija, **sin usar `label`**, aplicada en este orden:

1. Toma los 30 casos de `demo_cases.csv` con `window_observations == 12` y `readouts_until_cut ≥ 20`.
2. **Requiere revisión**: entre los casos con `selected == True`, el de menor `rank`. En empate, el menor `vehicle_id` numérico.
3. **Punto medio**: entre los casos con `20 < rank ≤ 250`, el de `rank` más cercano a 50.
4. **No requiere**: el caso con `rank` más cercano a `round(total_grupo / 2)` de su fold.
5. Si algún criterio queda sin candidato entre los 30, aplica el mismo criterio sobre `oof_priority_queue.csv` (con los mismos filtros de ventana). En ese caso, `readouts_until_cut` se calcula con la telemetría.
6. Registra en un log qué regla eligió cada camión. Permite sobrescribir la selección con una lista de ids en la configuración.

## 6. Contrato del payload (`version: 2`)

Sigue la forma exacta de `ejemplo_payload.json`. No uses NaN ni Infinity: usa `null`.

### Raíz

| Campo | Valor |
|---|---|
| `version` | `2` |
| `modo` | `"real"` |
| `fuente` | `"SCANIA Component X · camiones de entrenamiento evaluados por grupos"` |
| `modelo` | `{"capas": [527, 64, 5], "extra_ventana": 2}`, leído de `selection.json` |
| `clases` | Los 5 textos cortos de las clases 0–4 (ver el ejemplo) |
| `sensores` | Igual que el ejemplo: A = 171_0, B = 666_0, C = 427_0, perfil = 397 |
| `estadisticas` | Las 5 estadísticas del vectorizador, **en su orden real**: `[{"clave": <nombre interno>, "nombre": <texto>}]`. Textos sugeridos: Último, Promedio, Variación, Tendencia, Faltantes. Si una clave contiene "falt", "miss" o "ausen", la plantilla la muestra como porcentaje. |
| `capacidad` | `selected_capacity` (20) |
| `grupos` | Número de folds (5) |
| `camiones` | Los 3 camiones, en el orden: requiere revisión, punto medio, no requiere |
| `flota` | Ver más abajo |
| `federacion` | `{"reportes_enviados": raw_records, "mb_compartidos": published_state_bytes/1e6, "mb_centralizado": centralized_feature_baseline_bytes/1e6}` (redondeados a 2 decimales) |

### Cada camión

| Campo | Cálculo |
|---|---|
| `id`, `nodo`, `grupo` | `vehicle_id`; `node_id` con mayúscula inicial; `fold` |
| `caso` | `"Requiere revisión"`, `"Punto medio"` o `"No requiere"` (solo para el presentador) |
| `sensores_totales` | 105 |
| `lecturas_totales` | `readouts_until_cut` |
| `lecturas` | Los últimos `max(20, window_observations)` reportes con `time_step ≤ time_step del caso`, ordenados. Cada uno: `{"n": número en la historia completa (desde 1), "t": time_step, "valores": {"A","B","C"} crudos o null, "perfil": [6] o null, "sin_dato": nº de variables None de las 105}` |
| `perfil` | Divide `397_0…397_35` en 6 rangos con `np.array_split` y calcula el incremento de cada rango respecto del reporte anterior de la historia completa, dividido por la suma de incrementos. Usa `null` si falta algún valor, si algún incremento es negativo o si la suma es ≤ 0. |
| `ventana` | `{"inicio": window_start_readout_index − lecturas[0].n, "fin": len(lecturas) − 1}`. Los índices son posiciones dentro de `lecturas`. |
| `resumen` | `{"A": {clave_estadística: valor}, "B": {…}, "C": {…}}`: las 5 estadísticas **sin escalar** que produce `WindowVectorizer` en esa ventana |
| `probabilidades` | `[probability_0, …, probability_4]`, las calibradas del CSV |
| `puesto` | `rank` |
| `total_grupo` | Nº de filas de la cola con el mismo `fold` |
| `prioritario` | `selected`. Debe cumplir `puesto ≤ capacidad`. |
| `umbral` | `calibrated_probability_4` de la fila con `rank == capacidad` en el mismo `fold` |
| `fila` | 100 medias de `np.array_split(calibrated_probability_4 del fold ordenadas de mayor a menor, 100)` |

**No incluyas `label`.** El opcional que revelaba la clase no fue elegido.

### `flota` (desde `policy_summary.csv`)

```json
{"vehiculos_por_grupo": 4710, "urgentes_por_grupo": 400, "promedio_de_grupos": 5,
 "capacidades": [{"revisiones": 20, "encontrados": 12.0, "azar": 1.7, "veces": 7.1, "cobertura": 0.03}, …]}
```

- `encontrados` = `precision_at_capacity_mean × capacity`
- `azar` = `precision_at_capacity_mean / priority_lift_mean × capacity`
- `veces` = `priority_lift_mean`
- `cobertura` = `recall_at_capacity_mean`
- `vehiculos_por_grupo` = media de filas por fold de la cola
- `urgentes_por_grupo` = número de filas con `label == 4` en la cola, dividido entre el número de folds y redondeado

Este es el único uso de `label`, y es agregado. Incluye las 4 capacidades (20, 50, 100 y 250).

## 7. Integración en Streamlit

```python
# minetwin/demo/payload.py
def construir_payload_demo(ids: list[str] | None = None) -> dict: ...

# minetwin/views/feria.py
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from minetwin.demo.payload import construir_payload_demo

PLANTILLA = Path(__file__).resolve().parents[1] / "demo" / "flujo.html"
MARCADOR = "/*__MINETWIN_PAYLOAD__*/null"

def html_con_payload(payload: dict | None) -> str:
    plantilla = PLANTILLA.read_text(encoding="utf-8")
    if payload is None:
        return plantilla  # muestra el ejemplo incrustado
    assert plantilla.count(MARCADOR) == 1
    js = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    return plantilla.replace(MARCADOR, js, 1)

@st.cache_data(show_spinner="Preparando la demo…")
def _html(firma: str) -> str:
    return html_con_payload(construir_payload_demo())
```

- Agrega "Demo feria" como **primera opción** del radio de navegación.
- Al entrar, oculta con CSS la barra lateral, el encabezado y el botón Deploy, y reduce el padding.
- Pon un botón "← Salir del modo feria" que cambie la vista mediante un callback `on_click` sobre la `key` del radio.
- Renderiza con `components.html(_html(firma), height=860, scrolling=False)`.
- `firma` = hash del `manifest.json` de `07_demostracion` más el de `policy.json`, para invalidar la caché.
- Si `construir_payload_demo` falla, muestra `st.warning` **visible encima de la demo**, con el motivo y el texto "Se muestra un ejemplo: los camiones 2 y 3 son ilustrativos", y renderiza `html_con_payload(None)`. La demo nunca debe caerse.
- Convierte los tipos NumPy y Pandas a tipos nativos, y NaN a `None`.

**Opcional, para funcionar sin internet:** reemplaza en tiempo de ejecución las 3 líneas `<link … fonts…>` por un `@font-face` con los `.woff2` de Source Sans 3 en base64. Es el único cambio permitido sobre el HTML.

## 8. Tests (pytest)

1. **Contrato**: un validador en Python equivalente a `validar()` de la plantilla. Debe pasar con el payload generado y con `ejemplo_payload.json`.
2. **Fidelidad**: las `probabilidades`, el `puesto` y `prioritario` de cada camión coinciden con su fila en `demo_cases.csv` (o en la cola).
3. **Umbral coherente**: `prioritario == (probabilidades[4] ≥ umbral)` para los 3 camiones.
4. **Causalidad**: ningún reporte de `lecturas` tiene `t` mayor que el `time_step` del caso, y `lecturas[ventana.fin].n == readouts_until_cut`.
5. **Sin etiqueta**: el JSON no contiene la clave `label` en ningún camión.
6. **Serialización**: `json.dumps(payload, allow_nan=False)` no falla y el marcador se reemplaza exactamente una vez.
7. **Opcional, con Playwright**: el HTML generado da `window.__MINETWIN_ESTADO.ok === true`.

## 9. Criterios de aceptación

- [ ] "Demo feria" aparece primero en la navegación y se ve a pantalla completa.
- [ ] Con datos reales no aparece el aviso de ejemplo de Streamlit.
- [ ] Los 3 camiones (5 etapas cada uno) y Toda la flota (2 etapas) se recorren sin errores, y la página no requiere scroll a 1366×768.
- [ ] El camión "Requiere revisión" muestra "Revisión humana prioritaria"; los otros dos, "No priorizado por ahora".
- [ ] La etapa 6 muestra 12 de 20 frente a ≈ 2 al azar, con los valores de `policy_summary.csv`.
- [ ] Si se borra un artefacto, aparece un aviso y el ejemplo, sin romperse.
- [ ] Todos los tests pasan y las demás vistas no cambian.

## 10. Lo que NO debes hacer

- Modificar el diseño, los textos o las animaciones de `flujo.html`.
- Reentrenar, recalibrar o volver a predecir. Solo lees artefactos.
- Usar `label` para elegir camiones o incluirlo en un camión.
- Decir o insinuar que la predicción del caso viene de FedProx u otro régimen federado: viene del MLP evaluado por grupos.
- Usar las métricas de validation o comparar la línea base con la política en la demo.
- Escribir dentro de `results/`.

## 11. Orden de trabajo

1. Fase 0 e informe. **Esperar confirmación.**
2. Copiar los archivos, crear la vista con el ejemplo y verificarla dentro de Streamlit.
3. `construir_payload_demo` con el camión "Requiere revisión" únicamente. Pasar los tests 1–6.
4. Agregar los otros 2 camiones, `flota` y `federacion`.
5. Caché, respaldo y botón de salida.
6. Revisar los criterios de aceptación y reportar con capturas de las 5 etapas de un camión y de las 2 de Toda la flota.
