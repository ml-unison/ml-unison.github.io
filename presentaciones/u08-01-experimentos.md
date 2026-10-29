---
theme: none
title: Experimentos reproducibles y registro de modelos
unidad: Unidad 8
layout: portada
subtitulo: "Qué se probó, con qué resultado, cuál se eligió y por qué: MLflow, versiones, alias y fichas de modelo"
fecha: 2026-2
canvasWidth: 1280
aspectRatio: 16/9
routerMode: hash
colorSchema: light
transition: fade
favicon: /marca/favicon.png
fonts:
  sans: Source Sans 3 Variable
  serif: Source Sans 3 Variable
  mono: JetBrains Mono Variable
  provider: none
highlighter: shiki
lineNumbers: false
browserExporter: true
drawings:
  persist: false
info: |
  Reconocimiento de Patrones · Unidad 8 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Experimentos reproducibles y registro de modelos

---
layout: cita
autor: D. Sculley et al., «Hidden Technical Debt in Machine Learning Systems», NeurIPS 2015
nota: Pie de la figura 1 del artículo, ingenieros de Google. La figura dibuja el código de aprendizaje como un cuadrito negro rodeado de configuración, recolección y verificación de datos, administración de procesos, servicio y monitoreo.
---

Solo una pequeña fracción de los sistemas reales de aprendizaje automático está compuesta por *el código de aprendizaje automático*. La infraestructura que lo rodea es vasta y compleja.

<!--
Abrir la unidad con esta idea: hasta ahora todo el curso ha sido ese cuadrito negro. Esta unidad es sobre lo que lo rodea. La «deuda técnica» del título es la metáfora de Cunningham: atajos que hoy ahorran tiempo y mañana cobran intereses.
-->

---
layout: pregunta
---

# ¿Podrías reproducir dentro de seis meses el mejor modelo que has entrenado?

<v-click>

¿Qué semilla usaste? ¿Qué versión de scikit-learn? ¿Qué columnas quitaste y en qué celda? ¿El CSV es el mismo? ¿Ejecutaste las celdas en orden? Si la respuesta depende de tu memoria, **no**.

</v-click>

<!--
Dejar que respondan. Casi todos dirán que sí al principio; las preguntas del clic los hacen dudar. Lo importante: no es un problema de disciplina personal, es que la información no quedó guardada en ningún lado.
-->

---

# El mismo código, diez números

<SemillasAUC :filas="$clicks < 1 ? 1 : 2" :referencia="$clicks >= 2" />

<div class="cols" style="gap: 40px; margin-top: 8px">
<div class="pequeno">

Riesgo de incumplimiento de tarjetas de crédito (Taiwán, 22,500 clientes). Bosque: 200 árboles, diez semillas, la misma partición de validación.

</div>
<div class="pequeno" v-click>

Logística: nada aleatorio en el modelo, pero cambia cómo se reparten los clientes en cinco pliegues. El rango es de **1.3 a 1.4 milésimas**.

</div>
</div>

<div v-click class="nota">

La franja dorada separa al primer y al segundo lugar de la búsqueda de hoy. **El ruido de la semilla es del mismo tamaño que la diferencia que usamos para elegir.**

</div>

<!--
Con la semilla fija, dos corridas dan exactamente el mismo número, bit por bit (lo comprobamos en la libreta). El problema no es el azar: es no saber qué azar se usó.
-->

---

# Cinco fuentes de no reproducibilidad

| fuente | ejemplo | remedio |
|---|---|---|
| azar | inicialización, *bootstrap*, pliegues, minilotes | fijar y **guardar** las semillas |
| versiones | un valor por omisión que cambia entre versiones | registrar el entorno; `uv.lock` |
| datos | alguien corrige un valor del CSV | guardar una **huella** (*hash*) |
| cómputo paralelo | sumas en punto flotante en otro orden (GPU, hilos) | modos deterministas, a cambio de velocidad |
| la libreta | celdas fuera de orden, variables de celdas borradas | reiniciar y ejecutar todo; un guion |

<div class="nota">

La suma en punto flotante no es asociativa: $(a + b) + c \ne a + (b + c)$ en general. Si mil hilos suman en el orden en que terminan, el resultado cambia en los últimos dígitos, y un árbol puede cortar en otro lado.

</div>

---

# Qué guardar de cada corrida

<div class="cols cols-4-6" style="gap: 48px">
<div>

1. **código**: el *commit*
2. **datos**: una huella
3. **entorno**: versiones
4. **parámetros**: hiperparámetros, semillas, partición
5. **métricas**: media **y** dispersión
6. **artefactos**: modelo, gráficas, configuración

</div>
<div>

Lo que guardamos en la libreta, como etiquetas de cada corrida:

```text
codigo.commit       2bd5a59
codigo.rama_curso   rediseno-2027
datos.nombre        credito_taiwan
datos.huella        3cf212053ed7be44
entorno.python      3.12.15
entorno.sklearn     1.9.1
entorno.lightgbm    4.7.0
entorno.mlflow      3.17.2
```

<div v-click class="pequeno">

Cambiar **un** límite de crédito de 30,000 clientes cambia la huella a `0d8bcea6f529f846`. La partición queda fija por la huella y la semilla 2011.

</div>
</div>
</div>

<!--
La huella es SHA-256 de pd.util.hash_pandas_object: cambia con un solo valor, y también si se reordenan las filas (ejercicio 3 de la libreta).
-->

---
layout: seccion
numero: 1
---

# Seguimiento de experimentos

MLflow: un lugar común para lo que antes vivía en la memoria.

---

# El vocabulario de MLflow

<div class="cols" style="gap: 48px">
<div>

- **Experimento**: un problema; agrupa corridas (`riesgo-credito`).
- **Corrida** (*run*): una ejecución del entrenamiento.
- **Parámetro**: lo que se decidió **antes**; se guarda como texto.
- **Métrica**: lo que se midió; un número, con paso opcional.
- **Etiqueta**: texto libre (*commit*, huella, versiones).
- **Artefacto**: un archivo (modelo, gráfica, JSON).

</div>
<div>

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///seguimiento.db")
mlflow.set_experiment("riesgo-credito")

with mlflow.start_run(run_name="bosque hoja≥50"):
    mlflow.set_tags(ENTORNO)
    mlflow.log_params({"n_estimators": 200,
                       "min_samples_leaf": 50})
    mlflow.log_metric("auc_media", 0.7810)
    mlflow.log_dict(config, "configuracion.json")
```

<div class="pequeno tenue">

En un equipo, el URI apunta a un servidor compartido; en tu máquina, `mlflow ui` muestra las corridas en el navegador.

</div>
</div>
</div>

---

# Una corrida, en código

<div class="cols cols-6-4 codigo-chico" style="gap: 40px">
<div>

```python
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=2011)

def correr(config):
    r = cross_validate(construir(config), X_tr, y_tr, cv=CV,
                       scoring=["roc_auc", "neg_log_loss"])
    with mlflow.start_run(run_name=config["nombre"]):
        mlflow.set_tags(ENTORNO)
        mlflow.log_params(config)
        mlflow.log_params({"cv.pliegues": 5, "cv.semilla": 2011})
        mlflow.log_metrics({
            "auc_media": r["test_roc_auc"].mean(),
            "auc_de":    r["test_roc_auc"].std(),
            "logloss_media": -r["test_neg_log_loss"].mean()})
        for paso, auc in enumerate(r["test_roc_auc"]):
            mlflow.log_metric("auc_pliegue", auc, step=paso)
        mlflow.log_dict(config, "configuracion.json")
```

</div>
<div>

<v-clicks>

- **La configuración es un diccionario**: todo lo necesario para reconstruir el modelo, y nada más.
- Se evalúa siempre con **la misma partición**, y la partición también se registra.
- Se guarda la **dispersión** entre pliegues, no solo la media.
- Los parámetros vuelven como texto (`'50'`); el JSON conserva los tipos.

</v-clicks>

</div>
</div>

---

# Siete corridas, una consulta

<CorridasComparadas :logloss="$clicks >= 1" />

<div class="cols" style="grid-template-columns: 1fr 1fr 1fr; gap: 28px; margin-top: 4px">
<div class="pequeno">

<span style="color: #D5433C; font-weight: 650">Árboles</span> y <span style="color: #A87C12; font-weight: 650">*boosting*</span> le ganan a la <span style="color: #24398A; font-weight: 650">logística</span> por unos 6 puntos de AUC.

</div>
<div class="pequeno" v-click>

La ***log-loss*** delata al *boosting* agresivo (tasa de aprendizaje η = 0.3, 63 hojas): sobreajusta y da probabilidades extremas.

</div>
<div class="pequeno" v-click>

El 1.º y el 2.º lugar difieren en **0.0017**, la mitad de la desviación entre pliegues: un empate técnico.

</div>
</div>

<!--
La tabla salió de mlflow.search_runs(order_by=["metrics.auc_media DESC"]), que devuelve un DataFrame. Cualquiera con acceso al servidor puede repetir la consulta; también con filtros: filter_string="params.familia = 'boosting' and metrics.auc_media > 0.775".
La logística pierde porque trata los códigos de atraso (−2, −1, 0, 1, …) como números en una escala; los árboles cortan en «pagó a tiempo» contra «se atrasó».
-->

---

# *Autolog*: cómodo, con letra chica

<div class="cols" style="gap: 48px">
<div>

```python
mlflow.sklearn.autolog(log_models=False)
with mlflow.start_run(run_name="autolog: logística"):
    modelo.fit(X_tr, y_tr)
```

<v-click>

Registró **40 parámetros**, también los que no elegimos:

```text
logisticregression__C          1.0
logisticregression__solver     lbfgs
logisticregression__max_iter   2000
```

</v-click>
</div>
<div>

<v-click>

Y estas métricas:

| métrica | valor |
|---|---|
| `training_roc_auc` | 0.722 |
| `training_log_loss` | 0.465 |
| `training_accuracy_score` | 0.810 |

</v-click>

<v-click>

<div class="nota">

El prefijo lo dice: son **de entrenamiento**, medidas con los datos del ajuste. *Autolog* sirve para no olvidar nada; la evaluación honesta la diseñamos nosotros.

</div>

</v-click>
</div>
</div>

---
layout: seccion
numero: 2
---

# El registro de modelos

¿Qué modelo está en uso, y cuál lo va a reemplazar?

---
clicks: 2
---

# Las versiones se quedan; los alias se mueven

<VersionesAlias :paso="$clicks" />

<div class="cols" style="gap: 40px">
<div class="pequeno">

Un modelo registrado (`riesgo-credito`) tiene **versiones** numeradas e inmutables, cada una ligada a la corrida que la produjo.

</div>
<div class="pequeno">

Un **alias** apunta a una versión. Promover, o regresar a la anterior, es mover un alias **sin tocar el código** del sistema que usa el modelo.

</div>
</div>

<!--
Clic 1: llega el bosque como retador (versión 2). Clic 2: después de compararlos en prueba, el alias champion se mueve a la versión 2. La versión 1 no se borra.
-->

---

# Campeón contra retador

<div class="cols" style="gap: 48px">
<div>

Por primera vez se usan los **7,500 clientes de prueba**:

| versión | AUC | *log-loss* |
|---|---|---|
| v1 logística · `@champion` | 0.733 | 0.461 |
| v2 bosque · `@challenger` | **0.787** | **0.426** |

<v-click>

El retador gana por mucho más que el ruido entre pliegues: **se promueve**.

</v-click>
</div>
<div v-click>

```python
cliente = mlflow.MlflowClient()
cliente.set_registered_model_alias(
    "riesgo-credito", "champion", 2)
cliente.delete_registered_model_alias(
    "riesgo-credito", "challenger")

# en cualquier otro sistema:
modelo = mlflow.pyfunc.load_model(
    "models:/riesgo-credito@champion")
modelo.predict(clientes_nuevos)
```

<div class="pequeno tenue">

`pyfunc` es una interfaz común: quien llama no necesita saber si es un bosque, una red o una logística.

</div>
</div>
</div>

<!--
Advertencia de método: si usáramos el conjunto de prueba para decidir entre muchos retadores, dejaría de ser una estimación honesta (ejercicio 4 de la libreta).
-->

---

# La firma del modelo

<div class="cols cols-6-4" style="gap: 40px">
<div class="codigo-chico">

```python
firma = infer_signature(X_tr.head(100),
                        modelo.predict_proba(X_tr.head(100)))
mlflow.sklearn.log_model(modelo, name="modelo", signature=firma,
                         registered_model_name="riesgo-credito")
```

```text
inputs:  ['limite': long, 'sexo': long, 'escolaridad': long,
          'estado_civil': long, 'edad': long, 'atraso_1': long,
          …, 'pago_6': long]            (23 columnas, todas requeridas)
outputs: [Tensor('float64', (-1, 2))]
```

</div>
<div>

La **firma** declara qué entra y qué sale. Se guarda con el modelo y se verifica en cada predicción.

<v-click>

Si al sistema que llama se le olvida una columna:

```text
MlflowException → Model is
missing inputs ['limite'].
```

</v-click>

<v-click>

<div class="nota">

Mejor un error ruidoso que una predicción silenciosamente equivocada.

</div>

</v-click>
</div>
</div>

---

# Reproducir una corrida seis meses después

<div class="cols" style="gap: 48px">
<div class="codigo-chico">

```python
original = mlflow.get_run(id_corrida)
original.data.tags["datos.huella"] == huella_datos(credito)  # True
original.data.tags["entorno.sklearn"] == sklearn.__version__ # True

config = mlflow.artifacts.load_dict(
    original.info.artifact_uri + "/configuracion.json")
cv = StratifiedKFold(5, shuffle=True, random_state=2011)
auc = cross_validate(construir(config), X_tr, y_tr,
                     cv=cv, scoring="roc_auc")["test_score"].mean()
```

</div>
<div>

<v-clicks>

- Primero se verifica que **datos y entorno** sean los mismos: si no, ya sabemos dónde buscar la diferencia.
- Luego se reconstruye el modelo desde la configuración guardada.

</v-clicks>

<div v-click>

| | AUC |
|---|---|
| registrado | 0.7810276203 |
| repetido | 0.7810276203 |

Diferencia: **0.0**.

</div>
</div>
</div>

---

# Cargar un modelo es ejecutar código

<div class="cols" style="gap: 48px">
<div>

```python
class ModeloSospechoso:
    def __reduce__(self):
        return (print, ("⚠ se ejecutó al CARGAR",))

datos = pickle.dumps(ModeloSospechoso())
pickle.loads(datos)       # solo cargar
```

```text
⚠ se ejecutó al CARGAR
```

<div class="pequeno">

`pickle` es un lenguaje de instrucciones, y una es «llama a esta función». Un atacante pondría `os.system(...)`.

</div>
</div>
<div>

<v-click>

**`skops`** guarda datos y **audita** los tipos al cargar. MLflow lo usa por omisión con scikit-learn:

| modelo | tipos que hay que revisar |
|---|---|
| logística | ninguno |
| bosque | `sklearn.tree._tree.Tree` |
| LightGBM | `Booster`, `LGBMClassifier`, `OrderedDict` |

</v-click>

<v-click>

<div class="nota">

Los árboles guardan índices de nodos que scikit-learn usa sin verificar: un archivo manipulado puede leer memoria ajena. Solo se declaran confiables los tipos que revisamos.

</div>

</v-click>
</div>
</div>

<!--
La documentación de Python lo dice explícitamente: nunca deserializar con pickle datos de una fuente no confiable. Esto incluye modelos bajados de internet. joblib usa pickle por dentro.
-->

---
layout: seccion
numero: 3
---

# La ficha del modelo

Lo que el registro no dice: para qué sirve, para qué no, y con quién se equivoca.

---

# *Model cards*

<style>
.compacta li { margin-bottom: 2px !important; font-size: 22px; }
.compacta ol > li::before { width: 28px; height: 28px; line-height: 28px; font-size: 15px; top: 0; }
</style>

<div class="cols" style="gap: 48px">
<div>

M. Mitchell et al., «Model Cards for Model Reporting», FAT\* 2019: cada modelo publicado se acompaña de una ficha breve con nueve secciones.

<div class="compacta">

1. detalles del modelo
2. uso previsto **y usos fuera de alcance**
3. factores: grupos y condiciones relevantes
4. métricas
5. datos de evaluación
6. datos de entrenamiento
7. análisis cuantitativos, **desagregados por grupo**
8. consideraciones éticas
9. advertencias y recomendaciones

</div>
</div>
<div>

<v-click>

<Caja tipo="idea">

La ficha la lee **quien decide**, no quien entrena. Su aporte central es la sección 7: el promedio puede esconder que el modelo trata distinto a distintos grupos.

</Caja>

</v-click>

<v-click>

<div class="pequeno">

En la libreta la ficha se **genera desde el código**, para que sus cifras siempre coincidan con el modelo, y se guarda como artefacto de la corrida de la versión campeona.

</div>

</v-click>
</div>
</div>

---

# La ficha de `riesgo-credito`, versión 2

<div class="cols cols-6-4" style="gap: 40px">
<div class="tabla-ficha">

Prueba, umbral $t^\ast = 1/6$ sobre la probabilidad de incumplir (un incumplido no detectado cuesta 5 veces una alerta innecesaria). TVP y TFP: tasas de verdaderos y de falsos positivos.

| grupo | clientes | prevalencia | AUC | TVP | TFP |
|---|---|---|---|---|---|
| todos | 7,500 | 0.221 | 0.787 | 0.768 | 0.375 |
| hombres | 2,969 | 0.250 | 0.776 | 0.769 | 0.392 |
| mujeres | 4,531 | 0.202 | 0.793 | 0.767 | 0.365 |
| menores de 30 | 2,420 | 0.229 | 0.788 | 0.769 | 0.385 |
| 30 a 45 | 3,916 | 0.208 | 0.779 | 0.746 | 0.357 |
| mayores de 45 | 1,164 | 0.249 | 0.805 | 0.828 | 0.420 |

</div>
<div class="pequeno">

<v-clicks>

- **Uso previsto:** priorizar contactos preventivos a clientes existentes.
- **Fuera de alcance:** negar crédito a solicitantes nuevos; otros países o épocas.
- **Ética:** usa sexo, edad y estado civil. Un cliente cumplido mayor de 45 es señalado con probabilidad 0.42; uno de 30 a 45, con 0.36.
- **Advertencias:** datos de la crisis de 2005; probabilidades sin calibrar; empate técnico con el *boosting*.

</v-clicks>

</div>
</div>

<style>
.tabla-ficha td, .tabla-ficha th { padding-top: 6px; padding-bottom: 6px; }
</style>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. El ruido de una semilla puede ser del tamaño de la diferencia con la que elegimos un modelo.
2. De cada corrida: **código, datos, entorno, parámetros, métricas, artefactos**.
3. MLflow: experimentos, corridas y una consulta para compararlas.
4. Registro: **versiones** inmutables y **alias** que se mueven; la **firma** valida la entrada.
5. Un modelo cargado es código ejecutado; y un modelo sin ficha no dice para qué **no** sirve.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Seguimiento de experimentos con MLflow* (Unidad 8, ejemplo 1).

**Laboratorio 8:** un servicio de predicción vigilado.

**Lectura:** D. Sculley et al., «Hidden Technical Debt in Machine Learning Systems», NeurIPS 2015; M. Mitchell et al., «Model Cards for Model Reporting», FAT\* 2019; documentación de MLflow: *Tracking* y *Model Registry*.

</div>
</div>
</div>
