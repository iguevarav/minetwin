# MineTwin

MineTwin estudia mantenimiento predictivo con datos históricos públicos de SCANIA Component X. El proyecto prepara ventanas causales, entrena modelos locales, centralizados y federados, publica únicamente estados de riesgo y registra la trazabilidad de cada artefacto.

El alcance se limita a datos históricos públicos y no formula afirmaciones sobre operaciones mineras reales. Las particiones `alpha`, `beta` y `gamma` son divisiones experimentales de una fuente pública común.

## Datos requeridos

Ubica estos ocho archivos sin cambiar sus nombres en `minetwin/data/scania/`:

- `train_operational_readouts.csv`
- `train_specifications.csv`
- `train_tte.csv`
- `validation_operational_readouts.csv`
- `validation_specifications.csv`
- `validation_labels.csv`
- `test_operational_readouts.csv`
- `test_specifications.csv`

## Flujo reproducible

Cada salida debe usar un directorio nuevo.

```powershell
$dataset = "minetwin\data\scania"
python -m minetwin --prepare-scania $dataset --output results\phase1_scania
python -m minetwin --prepare-scania-replay $dataset --phase1 results\phase1_scania --output results\phase1_replay
python -m minetwin.learning prepare --dataset $dataset --phase1 results\phase1_scania --output results\phase2_cache --window-size 12 --stride 1
python -m minetwin.learning select --cache results\phase2_cache --output results\final_selection --folds 5 --seed 42
python -m minetwin.study learning --cache results\phase2_cache --selection results\final_selection --output results\final_learning_verified
python -m minetwin.learning publish --cache results\phase2_cache --models results\final_learning_verified\seed_100 --output results\final_federation_verified --split validation --regime fedprox
python -m minetwin.learning diagnose --cache results\phase2_cache --models results\final_learning_verified\seed_100 --output results\final_diagnostics_verified
python -m minetwin.learning interpret --cache results\phase2_cache --models results\final_learning_verified\seed_100 --output results\final_interpretability_verified --split validation --regime fedprox
```

La aplicación se inicia con:

```powershell
python -m streamlit run app.py
```

## Alcance

La evaluación predictiva usa los splits oficiales de SCANIA. La validación cruzada agrupada, la selección de hiperparámetros, las diez semillas, las métricas, las pruebas de Wilcoxon con corrección Holm y la importancia por permutación se generan a partir de esos datos.

La federación preserva las variables de cada partición dentro del proceso experimental y comparte estados publicados y parámetros de modelo. No representa contratistas independientes ni un despliegue distribuido real.

Langflow es una capa explicativa opcional. No modifica la predicción, no diagnostica una pieza física y no ejecuta mantenimiento.
