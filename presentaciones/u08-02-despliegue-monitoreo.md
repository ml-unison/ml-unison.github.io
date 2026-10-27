---
theme: none
title: Despliegue y monitoreo
unidad: Unidad 8
layout: portada
subtitulo: "Servir un modelo, vigilar el cambio y decidir cuándo reentrenar"
fecha: 2027-1
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

# Despliegue y monitoreo

---
layout: caso
caso: Caso · Un modelo que compraba casas
cifra: 304 <small>mdd</small>
cifraNota: "millones de dólares: reducción del valor del inventario de casas de Zillow Offers en el tercer trimestre de 2021."
fuente: Zillow Group, resultados del tercer trimestre de 2021 y carta a los accionistas (2 de noviembre de 2021).
---

# Zillow Offers, 2018–2021

Zillow, uno de los grandes portales inmobiliarios de Estados Unidos, compraba casas con ofertas calculadas por sus modelos de precios y las revendía meses después.

<v-click>

En noviembre de 2021 anunció el cierre del negocio y el recorte de cerca de **25 % de su personal**. Su director, Rich Barton: la imprevisibilidad de los precios de las casas «excede por mucho lo que anticipamos».

</v-click>

<v-click>

El modelo se validó con un mercado; operó en otro. Y la «etiqueta» de cada compra, el precio de reventa, llegaba meses después.

</v-click>

<!--
Cifras del comunicado de Zillow del 2 de noviembre de 2021 (8-K ante la SEC): reducción de inventario de 304.4 millones de dólares en el tercer trimestre, más 240–265 millones esperados en el cuarto; recorte aproximado de 25 % de la plantilla. En el informe anual de 2021 la reducción total del año fue de 407.9 millones. No conviene atribuirle todo «al algoritmo»: también hubo problemas de operación (contratistas, mano de obra) y una decisión de negocio de comprar más rápido. La lección para esta clase es la parte del modelo: validarlo no garantiza nada cuando el mundo cambia, y las etiquetas tardías ocultan el problema.
-->

---
layout: pregunta
---

# Tu modelo de rentas de bicicletas tiene en validación un error de 47 bicicletas por hora. Lo pones en producción el 1 de enero de 2012. ¿Cuánto error tendrá en octubre?

<v-click>

164 bicicletas por hora, y subestimando la demanda en 42 %. Y ningún monitor de las entradas lo habría notado.

</v-click>

---

# El problema de hoy: un año en producción

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<RentasAnios v-if="$clicks < 1" :futuro="false" />
<RentasAnios v-else />

</div>
<div>

Capital Bikeshare, Washington, D. C. Rentas por hora con calendario y clima (Unidad 4).

- *Boosting* con pérdida de Poisson, entrenado con 2011: hora, día laboral, día de la semana, clima, temperatura, humedad, viento.
- Validación en 2011: RECM **47.4**, sesgo +0.8 %.

<v-click>

<div class="nota">

2012, que el 1 de enero aún no existía: **65 % más rentas**. El sistema creció, y ningún atributo mide eso.

</div>

</v-click>
</div>
</div>

---
layout: seccion
numero: 1
---

# Servir un modelo

Un modelo que nadie puede consultar no sirve de nada.

---

# Tres formas de servir

| forma | cuándo | en las bicicletas | importa |
|---|---|---|---|
| **por lotes** | las predicciones se pueden calcular antes | cada noche, la demanda de mañana hora por hora, para redistribuir bicicletas | rendimiento, costo |
| **en línea** (API) | la entrada se conoce hasta el momento | la app: «¿habrá bicicletas aquí en 30 minutos?» | **latencia**, disponibilidad |
| **en el dispositivo** | sin red, o los datos no deben salir | el teclado que sugiere la siguiente palabra | tamaño, energía |

<div v-click class="nota">

Una API también recibe lotes. En la libreta, una predicción individual cuesta milisegundos; en un lote de 500, **cientos de veces menos** por predicción: casi todo es costo fijo (validar, serializar, llamar al modelo).

</div>

---

# Una API con FastAPI

<div class="cols codigo-chico" style="grid-template-columns: 11fr 9fr; gap: 36px">
<div>

```python
class Hora(BaseModel):
    fecha: dt.date
    hora: int = Field(ge=0, le=23)
    laboral: int = Field(ge=0, le=1)
    dia_semana: int = Field(ge=0, le=6)
    clima: int = Field(ge=1, le=4)
    temp: float = Field(ge=-25, le=50)       # °C
    humedad: float = Field(ge=0, le=100)
    viento: float = Field(ge=0, le=100)      # km/h
class Prediccion(BaseModel):
    rentas: float
    version: str
@app.post("/predecir", response_model=Prediccion)
def predecir(h: Hora):
    X = pd.DataFrame([h.model_dump()])
    X["clima"] = X.clima.clip(upper=3)   # igual que al entrenar
    pred = modelo.predict(X[ATRIBUTOS])
    bitacora.append(X.assign(prediccion=pred))
    return Prediccion(rentas=pred[0], version=VERSION)
```

</div>
<div>

<v-clicks>

- El **esquema** (Pydantic) valida tipos y rangos *antes* del modelo: un árbol predice algo con 300 °C.
- El **mismo preprocesamiento** que en el entrenamiento. Si difiere, el modelo recibe otra cosa de la que aprendió.
- La **versión** en cada respuesta: nombre y huella SHA-256 del archivo.
- La **bitácora**: sin registro de entradas y predicciones no hay monitoreo.

</v-clicks>

</div>
</div>

---

# Probar el servicio sin levantarlo

<div class="cols codigo-chico" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

```python
cliente = TestClient(app)
pet = {"fecha": "2012-06-01", "hora": 17,
       "laboral": 1, "dia_semana": 5, "clima": 1,
       "temp": 27.0, "humedad": 50, "viento": 15}

r = cliente.post("/predecir", json=pet)
# 200 {'rentas': 549.3, 'version': 'bicis-2011-v1+…'}

r = cliente.post("/predecir",
                 json=pet | {"temp": 81.0, "hora": 25})
# 422
#   hora: Input should be less than or equal to 23
#   temp: Input should be less than or equal to 50
```

</div>
<div>

`TestClient` hace peticiones HTTP a la app **sin servidor**: es la base de las pruebas automáticas que corren con cada cambio de código.

<v-click>

```python
def test_servicio(cliente):
    assert cliente.post(...pet).status_code == 200
    assert cliente.post(...hora=24).status_code == 422
    assert cliente.post(...sin temp).status_code == 422
    assert r.json()["rentas"] >= 0     # Poisson
```

</v-click>

<v-click>

<div class="pequeno tenue">

En producción: `uvicorn servicio:app --port 8000`.

</div>

</v-click>
</div>
</div>

---

# El contenedor: el entorno viaja con el modelo

<div class="cols codigo-chico" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY servicio.py modelo.joblib ./
EXPOSE 8000
CMD ["uvicorn", "servicio:app",
     "--host", "0.0.0.0", "--port", "8000"]
```

</div>
<div>

<v-clicks>

- `FROM`: un sistema mínimo con Python.
- `requirements.txt` con **versiones fijas** (`scikit-learn==1.9.0`): un modelo guardado con una versión puede no cargarse en otra.
- `COPY`: el código y **el** modelo validado; su huella debe coincidir con la del registro.
- El mismo contenedor corre en tu laptop, en el servidor de pruebas y en producción.

</v-clicks>

</div>
</div>

<div v-click class="nota">

`joblib` y `pickle` **ejecutan código** al cargar: nunca cargues un modelo de una fuente que no controlas.

</div>

---
layout: seccion
numero: 2
---

# Qué cambia y cómo detectarlo

El monitoreo, antes de que lleguen las etiquetas.

---

# Tres maneras de que el mundo cambie

$P(\vx, y) = P(\vx)\,P(y \mid \vx) = P(y)\,P(\vx \mid y)$, y cada factor puede moverse solo. Azul: 2011; rojo: 2012.

<TresCambios :mostrar="$clicks + 1" />

<div class="cols" style="grid-template-columns: repeat(3, 1fr); gap: 26px; font-size: 20px; line-height: 1.35; margin-top: 20px">
<div>

**Cambio de covariables.** Marzo de 2012 fue caluroso: 18.4 °C contra 13.6 °C.

</div>
<div v-click>

**Cambio de concepto.** A la misma temperatura, muchas más rentas.

</div>
<div v-click>

**Cambio en $P(y)$.** Solo se ve con las etiquetas.

</div>
</div>

<!--
El cambio de prevalencia propiamente dicho es P(y) cambiando con P(x|y) fijo (una epidemia que aumenta la proporción de enfermos). Recordar la Unidad 3: la precisión depende de la prevalencia aunque TVP y TFP no cambien. Aquí P(y) cambia como consecuencia del cambio de concepto.
-->

---

# Por qué la diferencia importa

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Error esperado en $\vx$, con $f(\vx) = \E[y \mid \vx]$:

$$
e(\vx) = \big(f(\vx) - h(\vx)\big)^2 + \sigma^2(\vx) .
$$

<v-click>

Si las entradas pasan a seguir $Q$ en lugar de $P$:

$$
E_{\text{out}}(Q) = \E_{\vx \sim Q}\big[e(\vx)\big] = \E_{\vx \sim P}\Big[\tfrac{q(\vx)}{p(\vx)}\, e(\vx)\Big] .
$$

</v-click>
</div>
<div>

<v-click>

**Solo covariables:** $f$ no cambia; el error se **repondera**. Malo solo donde $p(\vx) \approx 0$: extrapolación.

</v-click>

<v-click>

**Concepto:** $f$ se mueve y aparece $(f_{\text{nueva}} - h)^2$, incluso donde había muchos datos. Ningún reponderado lo arregla.

</v-click>

<v-click>

<Caja tipo="cuidado" etiqueta="Retraso de etiquetas">

Bicicletas: la respuesta llega en una hora. Crédito: meses. Una casa de Zillow: cuando se revende. Mientras tanto, solo vemos las **entradas** y las **predicciones**.

</Caja>

</v-click>
</div>
</div>

---

# El índice de estabilidad poblacional (PSI)

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Cubetas en los deciles de la **referencia**; $p_k$ y $q_k$, proporciones de referencia y actuales:

$$
\text{PSI} = \sum_{k=1}^{K} (q_k - p_k)\ln\frac{q_k}{p_k} .
$$

<v-click>

Separando la suma:

$$
\text{PSI} = \underbrace{\sum_k q_k \ln\tfrac{q_k}{p_k}}_{D(q\|p)} + \underbrace{\sum_k p_k \ln\tfrac{p_k}{q_k}}_{D(p\|q)} ,
$$

la divergencia de **Kullback–Leibler simetrizada**: $\ge 0$, cero solo si $q = p$.

</v-click>
</div>
<div>

<v-click>

**Regla de la industria** (del crédito): menos de 0.1, estable; de 0.1 a 0.25, mirar; más de 0.25, cambio importante.

</v-click>

<v-click>

<Caja tipo="cuidado">

Son umbrales de costumbre, no de una teoría. No dependen del tamaño de la muestra… y deberían.

</Caja>

</v-click>

<v-click>

<div class="pequeno tenue">

Cubetas vacías: $\ln(q_k/p_k)$ explota; se reemplaza la proporción nula por un $\varepsilon$ pequeño ($10^{-4}$).

</div>

</v-click>
</div>
</div>

---

# El piso de ruido del PSI

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Con $q_k = p_k + \delta_k$ y $\ln(1 + \delta_k/p_k) \approx \delta_k/p_k$:

$$
\text{PSI} \approx \sum_k \frac{\delta_k^2}{p_k} \qquad (\text{una } \chi^2) .
$$

<v-click>

Sin **ningún** cambio, con $N$ datos de referencia y $M$ actuales, $\Var(\hat q_k - \hat p_k) = p_k(1 - p_k)\big(\tfrac1N + \tfrac1M\big)$, así que

$$
\E[\text{PSI}] \approx (K - 1)\Big(\frac1N + \frac1M\Big) .
$$

</v-click>
</div>
<div v-click>

Dos muestras de las **mismas** temperaturas de 2011, 300 repeticiones, $K = 10$:

| horas | PSI medio | fórmula | PSI > 0.1 |
|---|---|---|---|
| 72 (3 días) | 0.279 | 0.250 | 92 % |
| 168 (1 semana) | 0.115 | 0.107 | 57 % |
| 720 (1 mes) | 0.025 | 0.025 | 0 % |
| 8,000 | 0.002 | 0.002 | 0 % |

<div class="pequeno tenue">

Con una semana contra otra semana, la regla da falsa alarma más de la mitad de las veces.

</div>
</div>
</div>

---

# La trampa de la estacionalidad

<PSIEstacional v-if="$clicks < 1" modo="ingenuo" />
<PSIEstacional v-else />

<div class="cols" style="gap: 40px">
<div class="pequeno">

Contra **todo** 2011, cada mes de 2011 —los datos de entrenamiento— tiene PSI entre 1 y 5. Un monitor así suena siempre y nadie le hace caso.

</div>
<div class="pequeno" v-click>

Contra **el mismo mes** del año anterior, la temperatura solo resalta cuando de verdad fue distinta: marzo (1.21), enero, noviembre.

</div>
</div>

---

# El tablero: las entradas se mueven, las predicciones no

<MapaPSI :filas="5" />

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px; margin-top: 10px">
<div class="pequeno">

Cada mes de 2012 contra el mismo mes de 2011. Hay alarmas de temperatura, humedad y clima: es el tiempo atmosférico. Pero la fila de las **predicciones** apenas pasa de 0.1 en enero y marzo, cuando hizo más calor.

</div>
<div v-click class="pequeno">

**Conclusión de este tablero:** todo va bien. <span class="tenue">Guardemos esa conclusión.</span>

</div>
</div>

---

# Kolmogorov–Smirnov: sin cubetas

<div class="cols" style="grid-template-columns: 2fr 3fr; gap: 40px">
<div>

Compara las **distribuciones empíricas**:

$$
D = \sup_x \big|F_1(x) - F_2(x)\big| .
$$

<div class="pequeno tenue">

El supremo se alcanza en un dato: basta evaluar en la muestra combinada.

</div>

<v-click>

Sin cambio, la distribución de $D$ **no depende** de la de los datos; se rechaza la igualdad al 5 % si

$$
D > 1.358\sqrt{\frac{n + m}{nm}} .
$$

</v-click>
</div>
<div>

<KSDistribuciones />

<div class="pequeno tenue">

Marzo: 730 horas de 2011, 743 de 2012. $D = 0.457$, umbral 0.071, $p \approx 5 \times 10^{-70}$.

</div>
</div>
</div>

---

# Con muchos datos, todo es «significativo»

<KSTamano />

<div class="cols" style="gap: 40px; margin-top: 8px">
<div class="pequeno">

Temperaturas de 2011 contra las mismas más **0.5 °C**, un cambio que el modelo ni nota. $D$ se estabiliza en 0.04; el umbral baja como $1/\sqrt{n}$.

</div>
<div class="pequeno" v-click>

El valor $p$ pasa de 0.4 con 100 datos a $10^{-72}$ con 100,000. Con mucho tráfico, el umbral va sobre el **tamaño del efecto** ($D$, PSI), no sobre $p$.

</div>
</div>

<!--
Además, KS supone independencia y las horas consecutivas están muy correlacionadas (el clima de las 3 y de las 4 de la tarde es casi el mismo): los valores p son todavía más optimistas de lo que deberían.
-->

---
layout: seccion
numero: 3
---

# Cuando llegan las etiquetas

El error real y qué hacer con él.

---

# El error real: un cambio de concepto

<MapaPSI :filas="$clicks < 1 ? 5 : 6" />

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px; margin-top: 10px">
<div class="pequeno" v-click>

Rentas observadas **encendidas**, predicciones tranquilas: es la firma del cambio de concepto. RECM de 2012: **124.7** (validación: 47.4); el modelo subestimó todos los meses, **35.6 %** en el año.

</div>
<div class="pequeno" v-click>

Multiplicar todas las predicciones por **1.55** baja la RECM a 64.5: casi todo el daño es el **nivel** (el crecimiento), no la forma de la demanda. Pero ese 1.55 sale de las etiquetas de todo el año.

</div>
</div>

---

# Reentrenar: ¿con qué datos y cuándo?

<PoliticasReentreno />

<div class="cols" style="grid-template-columns: repeat(3, 1fr); gap: 28px; font-size: 19px; line-height: 1.35; margin-top: -6px">
<div v-click>

**Ventana de 12 meses** (promedio 89 contra 122): ayuda, pero mezcla el nivel de 2011.

</div>
<div v-click>

**Ventana de 3 meses** (71): se adapta al nivel; falla en marzo, cuando el calor le pide temperaturas que no vio.

</div>
<div v-click>

**Etiquetas 3 meses tarde** (103): en junio se predice con un modelo de enero a marzo. Otra estación.

</div>
</div>

<!--
Las cifras entre paréntesis son el promedio de las RECM mensuales de 2012. La moraleja no es «la ventana de 3 meses es la buena»: depende del retraso de las etiquetas y de la estacionalidad. Opciones intermedias para el ejercicio 4 de la libreta: pesos por antigüedad, o el modelo de 12 meses corregido por el cociente observado/predicho del último mes etiquetado.
-->

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 24px; line-height: 1.4">

1. Un servicio valida sus entradas, devuelve la **versión** del modelo y guarda una **bitácora**; el contenedor congela el entorno.
2. Puede cambiar $P(\vx)$, $P(y \mid \vx)$ o $P(y)$. Solo el cambio de concepto obliga a reaprender.
3. PSI = KL simetrizada; KS = máxima diferencia de distribuciones. Umbrales sobre el **tamaño del efecto**, con referencias que respeten las **estaciones** y conscientes del tamaño de muestra.
4. Las entradas no delatan el cambio de concepto: hay que medir el **error real** en cuanto lleguen las etiquetas.
5. Reentrenar es una política que se prueba contra el pasado, con el **retraso** de las etiquetas incluido.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Un servicio de predicción y la vigilancia del cambio* (Unidad 8, ejemplo 2).

**Laboratorio 8:** un servicio de predicción vigilado.

**Lectura:** C. Huyen, *Designing Machine Learning Systems*, O'Reilly (2022), caps. 7 a 9; J. Quiñonero-Candela et al. (eds.), *Dataset Shift in Machine Learning*, MIT Press (2009), cap. 1.

</div>
</div>
</div>
