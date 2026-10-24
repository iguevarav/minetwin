"""Domain explanations shared by screen views, downloads and reports."""

from dataclasses import replace

from minetwin.exports import Explanation

_SPECS = {
    "context": (
        "Contexto y trazabilidad",
        "Identidad de datos, modelo y evaluación activa.",
        "Verifique conjunto, vehículo, régimen y versiones antes de comparar resultados.",
        "Metadatos de los artefactos y de la selección activa; no son predicciones nuevas.",
        "La trazabilidad identifica la evidencia; no mide calidad predictiva.",
    ),
    "partitions": (
        "Distribución por partición",
        "Número de vehículos o ejemplos por nodo y conjunto.",
        "Compare volúmenes y participaciones; diferencias grandes indican heterogeneidad.",
        "Conteos del manifiesto, del índice de vehículos o de las ventanas causales, según las columnas.",
        "Un vehículo puede aportar varias ventanas; los nodos son particiones experimentales.",
    ),
    "classes": (
        "Distribución de clases",
        "Frecuencias de las cinco clases de Component X.",
        "Una clase dominante puede ocultar errores en clases escasas; consulte F1 macro y recall por clase.",
        "Conteo de etiquetas oficiales por conjunto; las ventanas de train pueden repetir vehículos.",
        "La frecuencia muestral no equivale a riesgo individual ni prevalencia en una mina.",
    ),
    "class_definitions": (
        "Significado de las clases",
        "Ventanas temporales hasta reparación definidas por SCANIA.",
        "La clase 4 corresponde a la mayor proximidad a reparación; lea cada intervalo de la tabla.",
        "Definiciones oficiales, sin transformación ni estimación del modelo.",
        "Component X es anónimo; la clase no identifica una pieza física.",
    ),
    "missing": (
        "Ausencia de datos",
        "Proporción de celdas sin observación antes de imputar.",
        "Menor proporción significa mayor cobertura; una ausencia no es un valor cero.",
        "Celdas ausentes divididas por readouts por variables del mismo conjunto.",
        "La cobertura no mide precisión predictiva y puede ocultar variables con más ausencias.",
    ),
    "separation": (
        "Separación descriptiva entre clases",
        "η² de las variables derivadas respecto a la clase.",
        "Un η² mayor indica más variación asociada con diferencias entre clases.",
        "Relación entre variación entre clases y variación total de cada variable en entrenamiento.",
        "La asociación descriptiva no demuestra causalidad ni dependencia del modelo.",
    ),
    "telemetry": (
        "Telemetría observada",
        "Evolución de variables anónimas y valores del readout actual.",
        "El eje horizontal es el paso temporal; los huecos representan ausencias. Compare cada señal en su propia escala.",
        "Lecturas originales hasta el corte activo; se conserva el orden temporal sin usar lecturas futuras.",
        "Las unidades y el significado físico de las variables anónimas no están disponibles.",
    ),
    "probabilities": (
        "Probabilidades por clase",
        "Probabilidad estimada por el modelo para las clases 0 a 4.",
        "Las probabilidades suman uno; la más alta puede diferir de la decisión que minimiza costo o de la prioridad.",
        "Probabilidades del artefacto indicado; solo son calibradas cuando así lo identifica la vista.",
        "Una probabilidad no confirma una falla ni explica una causa física.",
    ),
    "decision_cost": (
        "Costo esperado por decisión",
        "Penalización esperada de elegir cada clase.",
        "La decisión publicada minimiza el costo esperado; valores menores son favorables.",
        "Para cada decisión j se suma p(clase i) × costo SCANIA(i,j) sobre las cinco clases.",
        "El costo es una penalización del protocolo, no un importe monetario ni una orden de reparación.",
    ),
    "performance": (
        "Rendimiento predictivo",
        "Costo SCANIA y métricas de clasificación del régimen indicado.",
        "Menor costo; mayor F1 macro, exactitud balanceada y PR AUC son favorables. La exactitud sola oculta el desbalance.",
        "Métricas de los artefactos de evaluación; promedios y dispersión se calculan sobre las semillas o folds indicados.",
        "Comparar sobre la misma muestra; una diferencia de medias no demuestra superioridad estadística.",
    ),
    "calibration": (
        "Calibración de probabilidades",
        "Confianza del modelo y exactitud top-1 observada.",
        "Confianza media y exactitud próximas indican mejor calibración; Brier y ECE menores son favorables.",
        "Brier usa error cuadrático multiclase; ECE pondera diferencias de confianza y exactitud por intervalo top-1.",
        "La calibración top-1 no mide calibración de la decisión por costo; los intervalos pequeños son inciertos.",
    ),
    "confusion": (
        "Matriz de confusión",
        "Conteos por clase real (filas) y clase predicha (columnas).",
        "La diagonal son aciertos; fuera de ella hay confusiones. Compare con el soporte de cada clase.",
        "Se cruzan etiquetas oficiales y decisiones del modelo sobre el conjunto de evaluación indicado.",
        "Los conteos sin normalizar están influidos por el desbalance; no confundir con probabilidades.",
    ),
    "class_metrics": (
        "Métricas por clase",
        "Precisión, recall, F1, soporte y costo por clase.",
        "Recall mide la fracción detectada; precisión, la fracción correcta entre predicciones de esa clase. Valores mayores son favorables.",
        "Métricas derivadas de verdaderos positivos, falsos positivos y falsos negativos; F1 combina precisión y recall.",
        "Las clases con pocos casos tienen mayor incertidumbre; revise soporte y matriz de confusión.",
    ),
    "training": (
        "Pérdida durante el entrenamiento",
        "Valor de la función de pérdida por ronda y cliente o régimen.",
        "Una reducción sostenida describe el ajuste; oscilaciones señalan inestabilidad.",
        "Historial guardado durante el entrenamiento; las curvas por cliente promedian las observaciones de cada ronda cuando corresponde.",
        "Una pérdida menor en entrenamiento no garantiza menor costo ni mejor generalización en validación.",
    ),
    "selection": (
        "Selección de modelo o variante",
        "Configuraciones candidatas y costo medio de validación cruzada.",
        "Menor costo es favorable; las barras de error son desviación estándar, no intervalos de confianza.",
        "Evaluación interna agrupada por vehículo sobre train; los empates se resuelven con las reglas publicadas de selección.",
        "Los folds de selección no son la validación oficial y la búsqueda solo cubre los candidatos declarados.",
    ),
    "folds": (
        "Validación cruzada por vehículo",
        "Costo y métricas de cada fold del candidato elegido.",
        "Compare costo y dispersión; un costo menor indica menos penalización.",
        "Cada vehículo se mantiene íntegro en un fold; se evalúan vehículos retenidos dentro de train.",
        "Los folds reutilizan entrenamiento y no sustituyen una evaluación final independiente.",
    ),
    "statistics": (
        "Comparaciones estadísticas pareadas",
        "Diferencias entre regímenes, intervalo bootstrap, p ajustado Holm y efecto biserial.",
        "Antes de concluir superioridad, compruebe signo, intervalo que excluya cero y p ajustado menor que 0.05.",
        "Comparaciones sobre las mismas semillas; consulte la convención de diferencia de la tabla. Mejora positiva favorece al candidato; en costo, diferencia candidato−referencia negativa es favorable.",
        "Significancia no implica utilidad práctica; pocos pares limitan precisión y potencia.",
    ),
    "importance": (
        "Importancia por permutación",
        "Aumento del costo al permutar una variable o un grupo temporal.",
        "Un aumento positivo mayor indica más dependencia predictiva; valores negativos no son evidencia de aporte positivo.",
        "Se rompe la asociación del grupo con la etiqueta y se compara el costo con el costo base; DE resume repeticiones.",
        "No demuestra causalidad; variables correlacionadas pueden repartirse la importancia y las señales son anónimas.",
    ),
    "federation": (
        "Publicación y transferencia federada",
        "Registros privados, estados, mensajes y bytes compartidos.",
        "Distinga estados publicados de lecturas crudas; una cobertura mayor no significa mayor precisión.",
        "Estados medidos al serializar y parámetros estimados desde el tamaño del modelo; los registros crudos permanecen en su nodo.",
        "Comparación experimental en particiones lógicas; no es una medición de red ni validación entre organizaciones.",
    ),
    "queue": (
        "Cola de revisión prioritaria",
        "Posición y probabilidad de clase 4 de los vehículos de un mismo fold.",
        "Menor puesto significa mayor prioridad; se revisan los primeros K según la capacidad publicada.",
        "Orden descendente de probabilidad calibrada de clase 4, con predicciones fuera de fold; el corte es el puesto K.",
        "La prioridad orienta revisión humana y no confirma una falla ni activa mantenimiento.",
    ),
    "capacity": (
        "Capacidad de revisión de la flota",
        "Casos urgentes encontrados, referencia al azar y cobertura por capacidad.",
        "Compare encontrados con azar; mayor cobertura detecta más urgentes pero exige más revisiones.",
        "Encontrados = precisión@K × K; azar = encontrados / lift; cobertura = recall@K, promediados por fold.",
        "Es una evaluación fuera de fold en train; no garantiza el mismo rendimiento en una nueva flota.",
    ),
    "profile": (
        "Perfil de uso observado",
        "Participación de seis rangos del sensor anónimo 397.",
        "Cada rango expresa su fracción del incremento total; los perfiles válidos suman uno.",
        "Las 36 componentes se agrupan en seis rangos y se normalizan sus incrementos respecto del readout anterior.",
        "Se omite el perfil si faltan valores, hay incrementos negativos o el total no es positivo; no identifica un uso físico concreto.",
    ),
    "window": (
        "Resumen de la ventana causal",
        "Último valor, promedio, dispersión, tendencia y proporción ausente por señal.",
        "Compare los estadísticos de una misma señal; faltantes es una proporción y los demás conservan la escala del sensor.",
        "Estadísticos de WindowVectorizer sobre la ventana observada, sin escalar; la imputación causal conserva indicadores de ausencia.",
        "La tendencia es descriptiva; no explica causalmente una predicción.",
    ),
    "architecture": (
        "Arquitectura y procesamiento",
        "Dimensiones de entrada, capa oculta y salida de la red.",
        "Las 105 señales aportan cinco resúmenes cada una y se agregan dos variables de ventana: 527 entradas y cinco salidas.",
        "Arquitectura leída del artefacto de selección; el diagrama representa capas, no pesos ni activaciones reales.",
        "El caso demostrativo procede del MLP fuera de fold; el diagrama no atribuye la predicción a FedProx.",
    ),
}


def describe(kind: str, title: str | None = None, source: str | None = None):
    explanation = Explanation(*_SPECS[kind])
    if title:
        explanation = replace(explanation, title=title)
    if source:
        explanation = replace(explanation, source=source)
    return explanation
