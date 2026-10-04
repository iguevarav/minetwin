# Cierre de MineTwin con datos SCANIA

## Alcance

MineTwin usa exclusivamente SCANIA Component X. Los resultados predictivos, federados, de interpretabilidad y de Langflow deben proceder de los artefactos SCANIA verificables.

Los nodos son particiones lógicas de un dataset público único. No representan contratistas, operaciones mineras reales ni un despliegue federado distribuido.

## Secuencia de cierre

1. Generar `final_learning_verified` a partir de la caché causal y la selección.
2. Generar publicación FedProx, diagnóstico e interpretabilidad con la semilla 100 verificada.
3. Ejecutar Langflow con la definición exportada del flujo, el modelo identificado y revisión humana completa.
4. Generar `final_report_scania` y documentar sus resultados exactos.
5. Ejecutar controles de calidad y revisar la interfaz SCANIA.

## Reglas

- Toda salida derivada debe conservar hashes de entradas, configuración, split y modelo.
- Una vista o reporte debe rechazar artefactos incompatibles.
- No incorporar datos, métricas ni visualizaciones sintéticas.
- Mantener una implementación única para preparación causal, escalado, evaluación, publicación y contexto Langflow.
- Añadir pruebas de regresión para contratos y errores observables.
