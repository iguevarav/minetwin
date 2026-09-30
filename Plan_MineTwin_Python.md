# Mine Twin — Plan funcional de implementación

**Estado:** fases 1 a 4 implementadas, con evaluación cuantitativa y ejecución real de Langflow mediante Ollama. La fase 5 incorpora interfaz adaptable, instalación reproducible, README, documentación técnica, resultados y limitaciones. Permanecen únicamente las cuatro actividades excluidas expresamente: exportación del flujo, trazabilidad persistente de explicaciones, guion de demostración y ejecución de las verificaciones finales.

**Entrega:** prototipo académico reproducible en Python, con datos operativos simulados, almacenamiento en memoria e interfaz en español.

**Organización:** fases funcionales, sin calendario ni estimaciones de esfuerzo. Cada fase termina cuando cumple sus criterios de aceptación. Se puede pausar el desarrollo y retomarlo desde el último resultado comprobado.

## 1. Propósito y alcance académico

Mine Twin integrará telemetría heterogénea de camiones de acarreo simulados, mantendrá gemelos locales y permitirá observar, diagnosticar, estimar degradación y ejecutar mantenimiento con consecuencias en la operación.

El recorrido que debe demostrar es:

**Operación → sensores → normalización → actualización del gemelo → diagnóstico y pronóstico → mantenimiento → cambio observable en el activo.**

El activo simulado y su gemelo son objetos diferentes. El primero conserva el estado real del experimento; el segundo estima el estado a partir de observaciones. La validación se realiza mediante simulación, sin conexión con camiones físicos.

### Pregunta de investigación propuesta

¿Qué efecto tiene incorporar tendencias de degradación, frente a usar únicamente umbrales, sobre las fallas y la disponibilidad de una flota heterogénea simulada integrada mediante gemelos locales?

La interoperabilidad y la continuidad ante desconexiones se evalúan como propiedades del framework. La comparación entre políticas de mantenimiento constituye el experimento principal. No se presupone que la política predictiva sea superior.

El tema de referencia puede conservarse: *A Federated Digital Twin Framework for Predictive Maintenance of Heterogeneous Haul Truck Fleets in Open-Pit Operations*. El resumen, los objetivos y las conclusiones deben precisar que se trata de una federación lógica validada mediante simulación.

### Significado de federación

- Tres nodos conservan sus datos y gemelos locales.
- Cada nodo adapta su telemetría a un contrato común y calcula sus diagnósticos localmente.
- Un coordinador reúne las últimas vistas y consulta el historial normalizado disponible.
- La desconexión de un nodo representa pérdida de comunicación con el coordinador. Su activo y su gemelo local continúan funcionando.
- Todo se ejecuta en un proceso. No se pretende demostrar distribución física, seguridad entre organizaciones ni rendimiento de una red real.

El proyecto base no incluye aprendizaje federado. Este requeriría entrenamiento local y agregación de actualizaciones de modelos, además de un protocolo experimental específico.

## 2. Alcance de la primera entrega completa

| Elemento | Decisión |
|---|---|
| Flota | Nueve camiones ficticios, tres por perfil |
| Heterogeneidad | Tres perfiles con diferencias explícitas de capacidad, comportamiento y valores de referencia |
| Nodos | Alpha con JSON, Beta con XML y Gamma con CSV, procesados en memoria |
| Motor | Diagnóstico, tendencia y tiempo estimado hasta umbral; evaluación experimental pendiente |
| Frenos | Detección de temperatura anómala contextualizada por carga y pendiente |
| Neumáticos | Detección de pérdida persistente de presión, identificando la rueda afectada |
| Otros subsistemas | Fuera de la primera entrega |
| Analítica | Reglas explicables y extrapolación de un indicador observable |
| Langflow | Explicación en español de evidencia observada por camión, solicitada por el usuario |
| Mantenimiento | Órdenes básicas, parada, duración e intervención sobre un componente |
| Interfaz | Flota y detalle, Federación, Mantenimiento |
| Experimentos | Ejecución sin interfaz y exportación automática de resultados |
| Almacenamiento operativo | RAM con límites de retención |
| Evidencias | Archivos de configuración, resultados, eventos y figuras |

Los perfiles se identificarán como ficticios. No hace falta usar modelos comerciales exactos. Los esquemas JSON/XML/CSV tampoco se presentarán como formatos oficiales de fabricantes.

### Funcionalidades pospuestas

Escena 3D, modelos CAD, fichas de subsistemas sin analítica, gestión de repuestos, usuarios y permisos, base de datos, API propia, distribución de nodos en servicios separados, conexión a equipos reales y entrenamiento de modelos de aprendizaje automático. Langflow sí forma parte del alcance como servicio de explicación; no requiere aprendizaje federado, RAG ni una base vectorial.

Solo se incorporará una ampliación cuando responda a un requisito académico o a una limitación observada. No es necesaria para dar por completado el alcance base.

## 3. Base técnica y organización del código

Se conserva la base tecnológica del plan original:

| Herramienta | Uso |
|---|---|
| Python | Lógica, simulación y experimentos; fijar una versión compatible al iniciar |
| Streamlit | Interfaz y controles de demostración |
| Plotly | Series temporales y comparaciones |
| NumPy | Generación reproducible y cálculo numérico |
| pandas | Tablas y exportaciones cuando simplifique su implementación |
| dataclasses y Enum | Entidades y estados |
| Pydantic | Validación en la entrada de los adaptadores |
| Langflow | Flujo explicativo externo conectado mediante un cliente HTTP de la biblioteca estándar |
| pytest y Ruff | Verificación del comportamiento y consistencia del código |
| pyproject.toml y archivo de dependencias resueltas | Instalación reproducible |

### Estructura implementada

```text
app.py
minetwin/
    __main__.py        # Ejecución por consola
    domain.py          # Entidades e invariantes
    simulation.py      # Activos, reloj y escenarios
    telemetry.py       # Validación del contrato canónico
    adapters.py        # Conversión de JSON, XML y CSV
    quality.py         # Calidad, duplicados y tardíos
    analytics.py       # Diagnóstico del motor
    components.py      # Diagnóstico de frenos y neumáticos
    prediction.py      # Tendencia del motor
    maintenance.py     # Órdenes y efectos de las intervenciones
    federation.py      # Nodos y coordinación
    services.py        # Operaciones que utiliza la interfaz
    experiments.py     # Ejecución sin interfaz y exportaciones
    langflow_client.py # Contexto observado y acceso al flujo
    assistant_ui.py    # Consulta explícita de explicaciones
    ui.py              # Navegación y controles generales
    federation_ui.py   # Flota, nodos y trazabilidad
    maintenance_ui.py  # Órdenes y pronóstico
    visuals.py         # Mapa y gráficos
    styles.css         # Identidad visual
tests/
Plan_MineTwin_Python.md
```

Los límites de retención se aplican en los servicios locales. No se necesita una capa adicional de almacenamiento. El empaquetado y la configuración reproducible del entorno se dejan indicados para la entrega; no se realizan instalaciones durante la implementación.

### Reglas de diseño

- La lógica de negocio y simulación no depende de Streamlit, Plotly ni pandas.
- La interfaz invoca operaciones de aplicación y no modifica directamente los repositorios.
- El predictor solo recibe observaciones, contexto permitido e historial; nunca recibe el activo con su desgaste oculto.
- Las dependencias se conectan explícitamente. No se necesita un contenedor de inyección de dependencias.
- Crear contratos e interfaces donde permitan sustituir una implementación o verificar un límite importante. Evitar una interfaz y un DTO nuevos para cada función.
- Las reglas y parámetros experimentales se mantienen en una configuración identificable, no dispersos entre las páginas.

## 4. Contratos mínimos y reglas comunes

### Datos esenciales

| Concepto | Contenido mínimo |
|---|---|
| Perfil | Identificador, capacidad, señales aplicables y parámetros de referencia |
| Camión | Identificador, perfil, nodo, estado operativo, horómetro y ciclos |
| Componente | Identificador dentro del camión, tipo e información de intervención |
| Paquete de telemetría | Identificador de evento, nodo, camión, versión de esquema, captura, recepción y lecturas |
| Lectura | Sensor, componente cuando corresponda, valor opcional, unidad, calidad y motivo |
| Gemelo | Última observación, estado estimado por componente, versión y frescura |
| Predicción | Instante, componente, resultado opcional, explicación, calidad y versión de reglas |
| Alerta | Camión, componente, regla, severidad, estado e instantes de apertura y cierre |
| Orden | Componente, motivo, intervención, estado, inicio, duración y finalización |
| Ejecución experimental | Configuración, semillas, versión del código y reglas, métricas y eventos |

No todos los conceptos necesitan una clase independiente desde la primera fase.

### Tiempo y reproducibilidad

- Un reloj simulado controla toda la evolución. La velocidad de la demostración solo modifica cuántos pasos se ejecutan por actualización.
- El paso inicial propuesto es un minuto simulado, configurable y usado de forma coherente en las ecuaciones.
- El horómetro aumenta cuando el camión opera según la definición del escenario; no aumenta durante reparación. La disponibilidad se mide sobre tiempo simulado programado.
- La RUL y la pendiente del predictor usan horas de operación. El tiempo de captura y el de recepción usan la misma escala temporal simulada.
- El tiempo real de procesamiento se mide por separado.
- Renderizar una página, cambiar un filtro o repetir una consulta no avanza el reloj.
- La configuración define si el horizonte es suficiente para observar fallas. Cualquier aceleración artificial de degradación se declara como supuesto.

### Calidad y retención

- No convertir lecturas ausentes en cero. Las señales no aplicables no cuentan como faltantes.
- Validar las señales requeridas por componente. Una lectura defectuosa no invalida automáticamente los componentes independientes que sí tienen datos suficientes.
- Un error en la identidad o estructura del paquete produce un rechazo trazable.
- Identificar duplicados por nodo e identificador de evento. Un paquete duplicado no actualiza dos veces el gemelo ni genera otra intervención.
- Los paquetes anteriores al estado actual se descartan de la analítica base y se contabilizan como tardíos. Su diagnóstico puede conservarse en un registro acotado.
- Configurar la caducidad de las observaciones. Si faltan datos esenciales o están caducados, no emitir una nueva predicción; conservar la anterior con su instante y condición desactualizada.
- Acotar historiales, crudos, alertas cerradas y registro de duplicados. Documentar la ventana de retención; no prometer deduplicación ilimitada.
- Conservar resolución suficiente para la ventana analítica. Reducir puntos únicamente para visualización cuando haga falta y preservar picos relevantes.
- Al reiniciar la sesión se pierde el estado operativo. Los experimentos completados permanecen en sus archivos de resultados.

## 5. Fases funcionales

### Fase 0 — Definir un experimento implementable

**Resultado:** una especificación breve que permita construir el primer caso sin inventar reglas durante la programación.

**Incluye:**

- Un perfil inicial y un camión con motor como componente principal.
- Diccionario de las señales mínimas: carga relativa, régimen operacional, temperatura de motor, vibración y horómetro.
- Modelo simple de degradación oculta y su efecto en las señales observables.
- Condición de falla del activo, indicador observable propuesto y condición de alerta.
- Duraciones y efectos de reparación preventiva y correctiva.
- Configuración de operación normal, degradación progresiva y falla sin intervención.
- Métricas principales y supuestos que limitan su interpretación.

**Criterios de aceptación:**

- Cada señal tiene nombre, unidad, procedencia y rango simulado documentados.
- Se distingue una temperatura elevada por operación de una desviación persistente asociada a degradación dentro del modelo.
- La condición de falla puede evaluarse por el simulador sin que el predictor acceda a su estado oculto.
- Existe un escenario suficientemente largo para alcanzar una falla y otro normal para observar falsas alarmas.

**Evidencia:** diccionario y configuración inicial. Las ecuaciones son supuestos del prototipo, no leyes físicas validadas.

### Fase 1 — Observar un camión simulado de forma reproducible

**Resultado:** primera demostración ejecutable de operación, sensores, adaptación y gemelo.

**Incluye:**

- Estructura mínima del proyecto e instalación reproducible.
- Ciclo: espera, carga, traslado cargado, descarga y retorno vacío.
- Reloj explícito, semilla y almacenamiento acotado.
- Un nodo JSON con conversión a telemetría canónica.
- Gemelo actualizado desde las lecturas y una regla de alerta persistente.
- Pantalla básica con estado, señales y explicación de la alerta.
- Controles de paso, pausa y reinicio; ejecución por pasos disponible sin interfaz.

Solo se simulan variables necesarias para producir o interpretar las señales. La posición y el mapa pueden esperar.

**Criterios de aceptación:**

- Misma configuración y semilla reproducen la trayectoria.
- El ciclo modifica coherentemente carga y señales; horómetro y ciclos se acumulan según lo definido.
- Cambiar filtros o refrescar la pantalla no genera eventos nuevos.
- Una lectura ausente se representa como ausente y no produce un estado saludable inventado.
- El escenario degradado produce una alerta explicable y el normal permite medir falsas alarmas.

**Evidencia:** prueba reproducible del recorrido y captura de la pantalla mínima.

### Fase 2 — Cerrar el ciclo de pronóstico y mantenimiento

**Resultado:** un camión cuyo estado cambia como consecuencia de una decisión de mantenimiento, con resultados medibles.

**Incluye:**

- Indicador observable de degradación del motor, corregido por el régimen operacional definido.
- Tendencia ajustada contra horas de operación y estimación hasta un umbral.
- Órdenes abiertas, en progreso, completadas y canceladas.
- Parada durante intervención, duración configurada y efecto sobre el componente seleccionado.
- Falla del activo con parada y reparación correctiva.
- Políticas automáticas por umbral y con pronóstico, usando las mismas operaciones que la intervención manual.
- Primera exportación del recorrido completo para verificar la evaluación antes de ampliar la flota.

**Reglas analíticas:**

- Usar persistencia e histéresis para abrir y cerrar alertas; evitar una nueva alerta en cada lectura.
- Definir el indicador `d`, su umbral `d_crit`, ventana mínima, pendiente mínima y criterio de estabilidad.
- Estimar `tiempo_hasta_umbral = (d_crit - d_actual) / pendiente` solo con condiciones válidas. Una observación válida que ya supera el umbral indica tiempo cero.
- Con historial insuficiente, datos caducados o tendencia inadecuada, devolver «no estimable» y un motivo.
- Una estimación fuera del horizonte útil se muestra como «fuera del horizonte», sin sustituirla por un límite que parezca una predicción exacta.
- Presentar el resultado como RUL del motor únicamente cuando se haya evaluado su relación con la falla simulada definida. Si la evidencia solo respalda llegada a un umbral de alarma, mantener ese nombre y limitar las conclusiones.
- No añadir un índice numérico de riesgo si solo repite la información del indicador y la alerta. Bastan estado, tendencia, tiempo estimado y explicación.

**Reglas de mantenimiento:**

- La orden abierta no detiene por sí sola al camión. La parada comienza al iniciar la intervención.
- Mientras se repara, el camión no transporta y su horómetro operativo permanece detenido.
- El efecto se aplica una sola vez cuando transcurre la duración configurada. La interfaz no puede completar anticipadamente una intervención.
- La reparación parcial reduce desgaste; la sustitución lo restablece según configuración. Ninguna restaura otros componentes.
- En la versión base se cancela una orden antes de comenzar; no se modela la cancelación de una reparación ya iniciada.
- Conservar el historial y abrir una nueva ventana de tendencia después de intervenir. El gemelo refleja el resultado mediante el evento y las observaciones posteriores.

**Criterios de aceptación:**

- El predictor funciona sin acceso al desgaste oculto o al instante real de falla.
- Una trayectoria sin intervención alcanza la falla y permite comparar estimaciones previas con el resultado observado.
- La intervención consume tiempo, cambia únicamente el componente objetivo y no se aplica de nuevo al repetir una solicitud.
- Las políticas por umbral y con pronóstico producen resultados exportables bajo condiciones externas comparables.
- Se puede demostrar el recorrido completo: degradación, alerta, predicción, orden, parada, reparación y recuperación.

**Evidencia:** trayectoria, eventos de mantenimiento, error de pronóstico y tiempos de parada del primer caso.

### Fase 3 — Integrar la flota heterogénea y sus nodos

**Resultado:** nueve camiones integrados a través de tres nodos con autonomía local y trazabilidad.

**Incluye:**

- Tres perfiles ficticios y tres camiones por perfil.
- Adaptadores JSON, XML y CSV con diferencias explícitas de campos y unidades.
- Estado e historial locales; diagnósticos y pronósticos calculados en cada nodo.
- Coordinador que agrega vistas sin duplicar todo el historial crudo.
- Desconexión, reconexión, caducidad, duplicados, paquetes tardíos y entradas defectuosas.
- Detección de frenos exigidos y fuga de presión; intervenciones simples sobre esos componentes.
- Panel inicial para comparar dato de origen, dato normalizado y estado del nodo.

**Política de desconexión:** el activo y el gemelo local siguen avanzando. El coordinador conserva la última vista con su instante; no permite iniciar nuevas intervenciones remotas en ese nodo. Una reparación ya iniciada continúa localmente. Al reconectar se publica el estado actual; no se implementa una cola de reproducción de todos los paquetes pendientes.

El historial local retenido podrá consultarse al reconectar, sin alterar la política de publicación del estado actual.

**Indicadores de integración:** completitud de campos requeridos, validez de lecturas presentes evaluables, edad de la última observación, rechazos, duplicados y tardíos. Mostrar denominadores; sin datos evaluables, mostrar «sin datos». Las contradicciones entre señales se informan por separado.

**Criterios de aceptación:**

- Un mismo evento lógico en los tres formatos produce valores canónicos equivalentes, con tolerancia numérica explícita cuando corresponda.
- Las diferencias de perfil se aplican en las referencias de diagnóstico; no se compara únicamente temperatura absoluta entre camiones distintos.
- Desconectar un nodo envejece su vista central y no detiene los otros nodos ni su propio estado local.
- Reconectar recupera el estado actual sin retroceder a observaciones antiguas.
- Los errores de un paquete se aíslan y se registran sin detener el procesamiento de los demás nodos.
- La pérdida de una señal esencial impide únicamente las estimaciones que dependen de ella.
- Frenos y neumáticos muestran anomalías explicables sin atribuirles una RUL no evaluada.

**Evidencia:** equivalencia de adaptadores y secuencia de desconexión, continuidad local y recuperación.

**Implementado:**

- Alpha: TRUCK-001 a TRUCK-003, 150 t, JSON; Beta: TRUCK-004 a TRUCK-006, 220 t, XML; Gamma: TRUCK-007 a TRUCK-009, 180 t, CSV. Los perfiles tienen referencias propias de motor, frenos y presión.
- Doce lecturas requeridas por observación: carga, temperatura y vibración de motor, horómetro, temperatura de frenos, pendiente y seis presiones. Una señal ausente o inválida conserva su calidad y solo afecta a los análisis dependientes.
- XML expresa carga en toneladas, temperatura en kelvin, vibración en m/s y presión en psi. CSV usa porcentaje de carga, vibración en in/s y presión en bar. El contrato común usa carga relativa, °C, mm/s, horas, pendiente porcentual y kPa. Equivalencia numérica comprobada con tolerancia absoluta de 1e-8 en las pruebas.
- Caducidad central: 180 segundos por defecto, configurable mediante `PredictorConfig.stale_after_seconds`. Una desconexión invalida de inmediato la vigencia del pronóstico central aunque continúe el cálculo local.
- Retención predeterminada: 720 observaciones y eventos identificados por camión, 100 alertas cerradas, 100 órdenes cerradas y 1000 eventos de operación. El registro de identificadores del nodo se limita a la suma de las ventanas de sus camiones. Las métricas de calidad son acumulativas durante la sesión; la deduplicación no es ilimitada.
- Escenarios de sobretemperatura de frenos y fuga en rueda FL; diagnóstico de las seis ruedas. Persistencia de tres minutos; frenos comparados con referencia de carga y pendiente. Las reparaciones detienen el camión y afectan únicamente al componente indicado. No se modelan fallas físicas ni RUL de frenos o neumáticos.
- Vista Federación con conexión/desconexión, prueba de paquete inválido, duplicado y comparación entre origen y contrato común. Consulta de historial local únicamente con comunicación disponible.
- Demostración sin interfaz: `python -m minetwin --fleet --steps 180 --scenario normal`. La evaluación por lotes de la flota se completa en la fase 4.
- Cliente y panel Langflow implementados, sin llamadas automáticas. La instancia local y el modelo seleccionados se describen en la sección 6.

**Verificación de cierre:** 110 pruebas aprobadas mediante `python -m pytest -q -p no:cacheprovider`, Ruff sin errores y ejecución por consola de los nueve camiones con escenario de frenos. La interfaz se verificó con Streamlit AppTest. La conexión real con Langflow funciona en Playground; su evaluación por API queda registrada como cierre de evidencia de la fase 4. La revisión visual manual en distintos tamaños de pantalla corresponde a la fase 5.

### Fase 4 — Evaluar la flota y activar el asistente Langflow

**Resultado:** evidencia para responder la pregunta de investigación, junto con un asistente explicativo conectado y evaluado sobre datos observados.

**Estado implementado:** la matriz de calibración y evaluación está ejecutada para nueve camiones, tres escenarios, tres políticas y diez semillas por conjunto. Los resultados reproducibles, comparaciones pareadas, pruebas de interoperabilidad, continuidad y reporte HTML se encuentran en `results/phase4_calibration` y `results/phase4_evaluation`. La evaluación real con Langflow, Ollama, `llama3.2:latest` y `minetwin-explanation-2` produjo siete respuestas para los siete casos consultables y bloqueó correctamente el caso desconectado. `review.csv` queda disponible si se requiere una puntuación humana formal.

**Incluye:** ampliar el ejecutor de la fase 2 para comparar la flota por perfiles y semillas, separar ajuste y evaluación y exportar resultados reproducibles. Activar y validar el flujo Langflow cuando el usuario haya realizado su configuración externa.

| Experimento | Comparación | Métricas y evidencia |
|---|---|---|
| Interoperabilidad | Eventos equivalentes y defectuosos en los tres formatos | Equivalencia canónica, rechazos y trazabilidad |
| Continuidad | Nodo conectado, desconectado y reconectado | Frescura, continuidad local y de los demás nodos, recuperación |
| Diagnóstico de motor | Umbrales por perfil frente a reglas que además incorporan tendencia | Detección por episodio, falsas alarmas por hora operativa y anticipación |
| Pronóstico de motor | Estimación frente a falla observada sin mantenimiento preventivo | MAE en horas operativas, proporción de casos estimables y no estimables |
| Mantenimiento de motor | Política por umbral frente a política con pronóstico | Fallas, paradas, intervenciones y disponibilidad |
| Asistente Langflow | Resumen generado frente a la evidencia y las reglas deterministas | Fidelidad de cifras, identificación del componente, tratamiento de ausentes, errores y tiempo real de respuesta |

**Trabajo funcional de Langflow:**

- Flujo mínimo: Chat Input → Prompt → modelo de lenguaje → Chat Output. El usuario elige proveedor/modelo y configura sus credenciales en Langflow.
- Entrada: observación normalizada del camión seleccionado, perfil, diagnósticos, pronóstico, alertas abiertas y órdenes registradas. Sin desgaste oculto, verdad experimental ni historial crudo completo.
- La evidencia enviada al modelo incluye un resumen determinista prioritario, estados compactos por componente y detalles únicamente para componentes que requieren atención. Las explicaciones de componentes normales no se repiten para evitar que una regla descrita se interprete como una anomalía observada.
- Salida: condición actual, señales que la sustentan, limitaciones e inspección sugerida, en español y con extensión breve. La respuesta no crea órdenes ni modifica reglas, predicciones o simulación.
- Evaluar casos de operación normal, motor degradado, frenos exigidos, fuga de presión, datos incompletos y mantenimiento. Comprobar también credenciales erróneas, flujo indisponible, respuesta vacía y timeout.
- Las vistas desconectadas o caducadas bloquean la consulta. Una respuesta corresponde a una observación concreta; se oculta al avanzar, cambiar la orden o reiniciar. Las explicaciones anteriores no se presentan como vigentes.
- Mantener separadas la evaluación determinista y la evaluación del modelo de lenguaje. Langflow no interviene en las políticas comparadas ni en sus métricas.

**Comparación justa:**

- Ajustar parámetros en trayectorias diferentes de las utilizadas para evaluar; no dividir filas vecinas entre ajuste y prueba.
- Mantener los mismos perfiles, demanda programada, condiciones externas y reglas de reparación entre políticas.
- Definir ambas políticas antes de la evaluación. Por ejemplo: alerta persistente para la política por umbral; la misma condición más RUL válida inferior a un margen para la política con pronóstico.
- Cuando no exista pronóstico válido, la política predictiva conserva la regla por umbral. No interpretar «no estimable» como ausencia de riesgo.
- Separar las fuentes aleatorias o generar previamente las condiciones externas. La misma semilla por sí sola no garantiza comparabilidad si una intervención cambia el orden de los sorteos.
- Usar inicialmente diez semillas de evaluación y ampliar si la variación observada impide interpretar el resultado. Reportar diferencias pareadas y dispersión, además de promedios.
- Incluir alguna degradación con evolución distinta de una recta y cambios de régimen; no hacer que simulador y predictor resuelvan exactamente la misma ecuación.
- Evaluar inicialmente la política sobre el motor. Mantener iguales o desactivar los escenarios de falla de otros componentes para no confundir su efecto.

**Definiciones de métricas:**

- Disponibilidad = tiempo disponible para operar / tiempo programado. La espera operativa cuenta como disponible; falla y reparación cuentan como indisponibles.
- Definir inicio y final de cada episodio de degradación para medir detección y evitar contar la misma falla muchas veces.
- Comparar RUL en instantes o intervalos operativos predefinidos. Reportar error por trayectoria y agregado, junto con la cobertura de estimaciones.
- Si una trayectoria no alcanza falla durante la observación, reportarla como censurada; no inventar una RUL real para ella.
- La menor cantidad de fallas se interpreta junto con el número de intervenciones y la disponibilidad, para no premiar una política que repara continuamente.

**Paquete mínimo de resultados por ejecución:**

```text
results/<run_id>/
    manifest.json      # Configuración efectiva, semillas, versiones y estado
    metrics.csv        # Métricas por trayectoria, camión o política
    events.csv         # Fallas, alertas e intervenciones necesarias para evaluar
    predictions.csv   # Estimaciones e instantes utilizados en la evaluación
```

El evaluador puede exportar verdad de referencia para calcular métricas; ese acceso no se comparte con el predictor. No es necesario conservar indefinidamente toda la telemetría en RAM. Una ejecución interrumpida debe identificarse como incompleta y no mezclarse con las finalizadas.

**Criterios de aceptación:**

- Cada resultado se puede reproducir con su configuración, semillas y versión de código.
- La ejecución no depende de mantener una página de Streamlit abierta.
- Las comparaciones conservan condiciones externas equivalentes y documentan los supuestos de operación y mantenimiento.
- Los resultados distinguen mejora, ausencia de mejora y limitaciones de estimación.
- Las pruebas de continuidad demuestran federación lógica; no se presentan como mediciones de tolerancia a fallos de infraestructura real.
- El flujo real explica los casos previstos sin ejecutar acciones; sus límites, errores y resultados de revisión quedan registrados. Las pruebas con respuestas HTTP simuladas no sustituyen esta validación.

**Evidencia:** matriz experimental ejecutada, archivos de resultados y figuras con unidades y tamaño de muestra.

### Fase 5 — Consolidar la experiencia y la entrega académica

**Resultado:** aplicación clara para demostrar el sistema y documentación suficiente para repetirlo.

**Estado implementado:** la interfaz dispone de estilos adaptables para escritorio, tableta y móvil; `pyproject.toml` fija Python y dependencias; `README.md` concentra instalación, operación, experimentos, diccionario de datos, arquitectura, métodos, resultados y limitaciones. No se ejecutaron pruebas ni revisión manual final por decisión del usuario.

| Vista | Contenido necesario |
|---|---|
| Flota y detalle | Estado de camiones, selección de unidad, señales, anomalías, pronóstico de motor y explicación |
| Federación | Estado de nodos, calidad, frescura, trazabilidad y ejemplo crudo frente a normalizado |
| Mantenimiento | Alertas, órdenes, inicio de intervención, duración restante e historial |
| Asistente dentro del detalle | Explicación bajo demanda vinculada a la observación seleccionada, con estado de conexión y límites claros |

**Criterios de presentación:**

- Diseño minimalista, profesional y moderno, con una identidad visual minera.
- Fondo oscuro grafito, tarjetas ligeramente más claras y acentos ámbar para acciones principales y selecciones.
- Navegación lateral, indicadores principales y un área central para el mapa y las alertas.
- Espacios amplios, tipografía legible, iconos consistentes y textos breves. Diagnósticos completos, calidad y trazabilidad se muestran bajo demanda.
- Verde para estados operativos normales, amarillo para advertencias y rojo para condiciones críticas o alertas prioritarias; siempre acompañados de etiquetas. Los datos ausentes se muestran en un estado neutral, sin sugerir normalidad.
- Gráficos sencillos, con unidades, referencias claras y colores coherentes con la interfaz. Las visualizaciones 3D son opcionales.
- Distribución adaptable: paneles y gráficos en columnas cuando hay espacio, apilados en pantallas pequeñas; navegación lateral plegable.
- Estilos, tipografía del sistema e ilustraciones del recorrido resueltos desde el código local, sin fuentes, mapas o servicios externos.

- Controles de iniciar, pausar, avanzar, reiniciar y seleccionar escenario.
- Mantener el reloj y el estado de sesión al navegar. La actualización periódica solo avanza la simulación mediante una operación explícita y única.
- Mostrar unidades, instante, calidad y condición «no estimable» o «desactualizado» donde corresponda.
- En gráficos, marcar umbrales e intervenciones; separar señales con unidades incompatibles.
- Mantener la explicación próxima al resultado: qué señal cambió, respecto a qué referencia y qué acción sugiere.
- El resumen incorpora un mapa 2D del ciclo. Mientras no exista telemetría espacial, se utiliza un esquema referencial explícitamente etiquetado, sin atribuir coordenadas reales al camión.
- Los experimentos se ejecutan mediante un comando documentado. Una página de experimentos es opcional.

**Verificación final:**

- Ejecutar pruebas críticas de simulación, calidad, adaptadores, pronóstico, mantenimiento y continuidad.
- Probar manualmente navegación, controles, reinicio y pérdida de comunicación.
- Medir memoria, procesamiento y respuesta de las vistas con la flota completa; corregir límites o demoras que afecten a la demostración.
- Repetir un experimento con el procedimiento documentado y comprobar sus métricas.
- Probar el recorrido completo con Langflow configurado: consulta, cambio de camión, avance, desconexión y mantenimiento; verificar que ninguna respuesta antigua aparezca como actual.
- Incorporar al registro de explicaciones el identificador de observación y las versiones del flujo, prompt y modelo utilizadas, sin guardar credenciales. Exportar la definición del flujo probada para poder reconstruirla.

**Entregables:**

- Código, configuración y dependencias reproducibles.
- README con instalación, aplicación, experimentos, reinicio y localización de resultados.
- Diccionario de datos y supuestos del simulador.
- Descripción de arquitectura, autonomía local y políticas de calidad.
- Método de diagnóstico, pronóstico y mantenimiento.
- Resultados obtenidos y figuras del informe.
- Limitaciones: datos sintéticos, fidelidad simplificada, nodos en un proceso, memoria operativa volátil y ausencia de validación industrial.
- Guion de demostración: operación normal, degradación, explicación, desconexión, reconexión, mantenimiento, recuperación y comparación experimental.
- Guía de configuración manual de Langflow, flujo probado y casos evaluados; distinguir el resultado determinista de la explicación generada y reconocer que el modelo puede producir errores.

Los documentos pueden agruparse para evitar duplicaciones. No hace falta un archivo independiente por cada punto.

**Criterio de finalización:** el flujo completo funciona, los resultados se reproducen, la interfaz permite explicarlos y las conclusiones se limitan a la evidencia obtenida.

## 6. Langflow: configuración actual y validación pendiente

Langflow se integra en el detalle del camión, mediante el panel «Asistente de explicación» de Resumen, Alertas y Mantenimiento. Se consulta explícitamente con la simulación pausada. La conexión pasa por `langflow_client.py`; el simulador, los adaptadores y los diagnósticos funcionan sin ese servicio.

La integración usa `POST /api/v1/run/{flow_id}`, autenticación `x-api-key` y entrada/salida de tipo chat, conforme a la [API oficial de ejecución de flujos](https://docs.langflow.org/api-flows-run). Mine Twin utiliza la biblioteca estándar de Python para esta llamada; no necesita instalar el paquete Langflow dentro de su proceso.

**Configuración comprobada:**

- Langflow 1.12.2 ejecutado en Docker mediante `http://127.0.0.1:7860`.
- Ollama accesible desde el contenedor mediante `http://host.docker.internal:11434`.
- Modelo seleccionado: `llama3.2:latest`.
- Flujo: `Mine Twin`, identificador `e986074a63904e01b011fd6ab0f5838f`.
- Conexión y respuesta verificadas desde Playground.

El flujo utilizado para la evaluación debe limitarse a Chat Input → Prompt → modelo → Chat Output. Si se conserva un componente Agent, se deben retirar las herramientas URL y Web Search para que la explicación dependa exclusivamente de la evidencia enviada por MineTwin y no ejecute acciones externas.

Para iniciar MineTwin y registrar el modelo seleccionado se definen estas variables en la misma terminal. La clave API de Langflow se crea en la instancia y no se guarda en el repositorio:

```powershell
$env:MINETWIN_LANGFLOW_URL = 'http://127.0.0.1:7860'
$env:MINETWIN_LANGFLOW_FLOW_ID = 'e986074a63904e01b011fd6ab0f5838f'
$env:MINETWIN_LANGFLOW_API_KEY = 'CLAVE_API_DE_LANGFLOW'
$env:MINETWIN_LANGFLOW_FLOW_VERSION = '1'
$env:MINETWIN_LANGFLOW_MODEL_ID = 'llama3.2:latest'
$env:MINETWIN_LANGFLOW_PROMPT_VERSION = 'minetwin-explanation-2'
$env:MINETWIN_LANGFLOW_TIMEOUT_SECONDS = '300'
python -m streamlit run app.py
```

Para una instancia remota se requiere HTTPS. El tiempo de espera del cliente es de 15 segundos por defecto y puede configurarse entre 0 y 600 segundos. La evaluación local con Ollama usa 300 segundos porque la generación depende del hardware disponible. Si falta la configuración o falla la conexión, el panel lo indica y el resto de Mine Twin sigue disponible. No guardar claves en código ni en el plan.

**Estado de validación:** cliente, contrato HTTP, ausencia de llamadas automáticas, errores y bloqueo de datos caducados están cubiertos por pruebas automatizadas. La ejecución real con `minetwin-explanation-2` obtuvo siete respuestas para los siete casos consultables y bloqueó correctamente el caso desconectado. Las siete respuestas respetaron el límite de palabras. La revisión humana permanece disponible en `review.csv`.

## 7. Cómo avanzar según la disponibilidad

- Implementar una porción funcional comprobable de la fase activa y registrar el siguiente paso concreto.
- Al terminar una sesión de desarrollo, actualizar una nota breve: qué funciona, qué se verificó y qué queda pendiente.
- Una fase completada conserva un escenario o prueba que permita detectar regresiones durante las siguientes.
- Se pueden preparar configuraciones y documentación de fases posteriores, pero su aceptación depende de los criterios de las fases anteriores.
- Si aparece una mejora opcional, registrarla fuera del alcance base. Si aparece un defecto que invalida una métrica o el flujo funcional, resolverlo antes de continuar.
- Mantener código limpio y comentarios solo cuando aclaren una decisión necesaria. Las instalaciones y configuraciones externas se dejan indicadas; no se ejecutan como parte de la implementación. La documentación se modifica cuando se solicite expresamente.

| Fase | Estado actual | Resultado para cerrar |
|---|---|---|
| 0. Experimento implementable | Implementada | Parámetros registrados y conjuntos de calibración y evaluación separados |
| 1. Camión observable | Implementada | Simulación y gemelo reproducibles con alerta; pruebas automatizadas aprobadas |
| 2. Pronóstico y mantenimiento | Implementada | Predicción hasta umbral, órdenes, paradas y exportación verificadas |
| 3. Flota y federación | Implementada | Nueve camiones, tres formatos, continuidad y diagnóstico por componente verificados |
| 4. Evaluación y Langflow | Implementada; matriz y ocho casos ejecutados | Completar `review.csv` si se requiere puntuación humana formal |
| 5. Experiencia y entrega | Implementada dentro del alcance autorizado | Pendientes excluidos: flujo exportado, trazabilidad, guion y verificaciones finales |

## 8. Referencias conceptuales y técnicas

- [NIST: Digital twins](https://www.nist.gov/digital-twins): marco conceptual sobre representación, predicción y apoyo a decisiones.
- [McMahan et al., 2017](https://proceedings.mlr.press/v54/mcmahan17a.html): distinción entre federación de gemelos y aprendizaje federado.
- [Streamlit: estado de sesión](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) y [fragmentos](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment): soporte de la demostración por sesión.

Estas referencias orientan el diseño; no validan las ecuaciones ni los parámetros de los camiones simulados.
