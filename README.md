# MineTwin

MineTwin analiza mantenimiento predictivo con el dataset público SCANIA Component X. Construye ventanas causales de telemetría, compara entrenamiento local, centralizado, FedAvg y FedProx, y conserva trazabilidad entre datos, escalador, modelos, publicación e informe.

Las particiones `alpha`, `beta` y `gamma` son divisiones experimentales de un único dataset público. El proyecto no afirma validar un despliegue en una mina ni una federación entre contratistas reales.

## Datos requeridos

Ubica estos ocho archivos CSV en `minetwin/data/scania/`:

- `train_operational_readouts.csv`
- `train_specifications.csv`
- `train_tte.csv`
- `validation_operational_readouts.csv`
- `validation_specifications.csv`
- `validation_labels.csv`
- `test_operational_readouts.csv`
- `test_specifications.csv`

## Estructura de resultados

| Directorio | Contenido |
| --- | --- |
| `01_datos` | Manifiesto, particiones y resumen del dataset. |
| `02_reproduccion` | Índice para recorrer la telemetría histórica. |
| `03_cache` | Ventanas causales, variables derivadas y metadatos. |
| `04_seleccion_modelo` | EDA y CV agrupada de los cuatro MLP. |
| `05_seleccion_variante` | Comparación de pérdida y decisión, además de referencias CV. |
| `06_aprendizaje` | Diez semillas de entrenamiento local, centralizado y federado. |
| `07_federacion` | Estados de riesgo publicados sin variables crudas. |
| `08_diagnostico` | Baselines, métricas y matrices de confusión de validation. |
| `09_interpretabilidad` | Importancia por permutación. |
| `10_reporte/articulo` | Tablas, figuras PNG a 300 dpi y procedencia del artículo. |

## Flujo reproducible

Cada salida requiere una carpeta de destino inexistente.

```powershell
$dataset = "minetwin\data\scania"
python -m minetwin --prepare-scania $dataset --output results\01_datos
python -m minetwin --prepare-scania-replay $dataset --phase1 results\01_datos --output results\02_reproduccion
python -m minetwin.learning prepare --dataset $dataset --phase1 results\01_datos --output results\03_cache --window-size 12 --stride 1
python -m minetwin.learning select --cache results\03_cache --output results\04_seleccion_modelo --folds 5 --seed 42
python -m minetwin.learning.variant_comparison cv --cache results\03_cache --output results\05_seleccion_variante --folds 5 --seed 42
python -m minetwin.learning.variant_comparison references --cache results\03_cache --output results\05_seleccion_variante --folds 5 --seed 42
python -m minetwin.learning.variant_comparison study --cache results\03_cache --selection results\05_seleccion_variante --output results\06_aprendizaje
python -m minetwin.learning publish --cache results\03_cache --models results\06_aprendizaje\seed_100 --output results\07_federacion --split validation --regime fedprox
python -m minetwin.learning diagnose --cache results\03_cache --models results\06_aprendizaje\seed_100 --output results\08_diagnostico
python -m minetwin.learning interpret --cache results\03_cache --models results\06_aprendizaje\seed_100 --output results\09_interpretabilidad --split validation --regime fedprox
python -m minetwin.learning.article_report --results results --output results\10_reporte\articulo
```

La selección interna usa cinco folds agrupados por vehículo y evalúa la última ventana de cada vehículo retenido. La configuración operativa seleccionada usa pérdida sin ponderación y mínimo costo esperado. `validation` se reserva para la evaluación final. HistGradientBoostingClassifier solo es una referencia dentro de CV; no se evalúa en `validation`.

## Interfaz y Langflow

```powershell
python -m streamlit run app.py
```

Langflow es opcional y solo genera explicaciones sobre estados de riesgo publicados. No modifica predicciones ni ejecuta acciones de mantenimiento.

## Verificación

```powershell
python -m ruff check minetwin tests
python -m pytest -q -p no:cacheprovider
```
