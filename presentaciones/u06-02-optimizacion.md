---
theme: none
title: "Entrenar redes: optimización y regularización"
unidad: Unidad 6
layout: portada
subtitulo: "Fashion-MNIST: minilotes, momento, Adam, dropout y la pregunta de cuándo parar"
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
  Reconocimiento de Patrones · Unidad 6 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Entrenar redes: optimización y regularización

---
layout: pregunta
---

# Una red de medio millón de parámetros y 10,000 imágenes. ¿Qué puede salir mal al entrenarla?

<v-click>

Que no aprenda nada (la tasa de aprendizaje es mala), que aprenda muy despacio o a saltos (un problema de **optimización**), o que aprenda **demasiado bien** las 10,000 imágenes (un problema de **generalización**). Hoy vemos las dos cosas.

</v-click>

<!--
Dejar que el grupo proponga. Casi siempre salen «sobreajuste» y «que no converja». La clase se organiza en esas dos mitades.
-->

---

# Fashion-MNIST

<MuestrasFashion :filas="4" :tam="72" />

<div class="cols" style="gap: 40px; margin-top: 22px">
<div class="pequeno">

70,000 fotos de prendas del catálogo de Zalando, 28×28 en grises, 10 clases balanceadas. Xiao, Rasul y Vollgraf (2017) lo propusieron como reemplazo de MNIST, que ya era demasiado fácil.

</div>
<div class="pequeno" v-click>

**Problema:** clasificar la prenda a partir de los 784 píxeles. Camiseta, camisa, suéter y abrigo comparten silueta; una camisa se distingue por el cuello y los botones.

</div>
</div>

---

# El experimento de hoy

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 44px">
<div>

- Perceptrón multicapa 784 → 512 → 256 → 10, ReLU, *softmax* + entropía cruzada: **535,818 parámetros**.
- **10,000** imágenes para entrenar, otras 10,000 para validar, las 10,000 oficiales de prueba al final.
- Minilotes de $B = 128$; 30 épocas = 2,370 pasos.

<div class="pequeno tenue" v-click="1">

Referencia: la regresión *softmax* de la Unidad 4 (sin capas ocultas) llega a **0.828** de exactitud de prueba.

</div>
</div>
<div class="codigo-chico">

```python
def entrenar_epoca(modelo, minilotes, optimizador):
    modelo.train()          # activa dropout
    for xb, yb in minilotes:
        optimizador.zero_grad()
        perdida = F.cross_entropy(modelo(xb), yb)
        perdida.backward()   # retropropagación
        optimizador.step()   # SGD, momento, Adam…

@torch.no_grad()
def evaluar(modelo, X, y):
    modelo.eval()           # apaga dropout
    logits = modelo(X)
    return (F.cross_entropy(logits, y).item(),
            (logits.argmax(1) == y).float().mean().item())
```

</div>
</div>

<!--
El bucle es el mismo para todos los experimentos de la clase; lo único que cambia es el optimizador y la red. Señalar train()/eval(): olvidarlo es un error clásico con dropout y normalización por lotes.
-->

---
layout: seccion
numero: 1
---

# Descenso por gradiente estocástico

Un gradiente ruidoso, pero barato.

---

# El gradiente de un minilote

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

$E(\vw) = \frac1N\sum_n \ell_n(\vw)$. En lugar de $\mathbf{g} = \nabla E$ usamos

$$
\mathbf{g}_{\mathcal{B}} = \frac1B \sum_{n \in \mathcal{B}} \nabla\ell_n(\vw), \qquad \vw \leftarrow \vw - \eta\, \mathbf{g}_{\mathcal{B}} .
$$

<v-click>

Con índices uniformes, $\E[\mathbf{g}_{\mathcal{B}}] = \mathbf{g}$: **insesgado**. Y con índices independientes,

$$
\E\,\norm{\mathbf{g}_{\mathcal{B}} - \mathbf{g}}^2 = \frac{\sigma^2}{B} .
$$

</v-click>

<v-click>

<div class="pequeno">

Con un solo ejemplo, el ruido es **75 veces** la señal. Con $B = 128$, del mismo orden. Aun así funciona: los errores de pasos sucesivos se promedian.

</div>

</v-click>
</div>
<div>

<RuidoMinilote />

<div class="pequeno tenue">

200 minilotes de cada tamaño, después de una época de SGD. Sin reemplazo aparece el factor $\frac{N-B}{N-1}$ (línea punteada).

</div>
</div>
</div>

<!--
B·E‖g_B − g‖² vale 171, 175, 179, 174, 170 y 155: constante salvo el factor de población finita en B = 1024. Ruido como regularizador implícito: impide asentarse en mínimos muy estrechos (Keskar et al., 2017).
-->

---

# La tasa de aprendizaje

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 36px">
<div>

<BarridoTasa />

</div>
<div>

Misma red, misma inicialización, mismos minilotes; solo cambia $\eta$.

<v-click>

$\eta = 0.001$: en 237 pasos la pérdida apenas baja de 2.28 a 2.10. Entre 0.1 y 0.5, rápido. Con $\eta = 1$, a los 24 pasos la pérdida es `NaN`.

</v-click>

<v-click>

<div class="nota">

Cerca del mínimo, el error en la dirección del valor propio $\lambda$ de la hessiana se multiplica por $1 - \eta\lambda$ en cada paso: **diverge si $\eta > 2/\lambda_{\max}$**. Regla práctica: la mayor tasa estable, un poco por debajo.

</div>

</v-click>
</div>
</div>

---
layout: seccion
numero: 2
---

# Momento y Adam

Recordar los pasos anteriores, y darle a cada parámetro su escala.

---

# El zigzag, otra vez

<ValleMomento :inicial="0" />

<div class="cols" style="gap: 40px; margin-top: 10px">
<div class="pequeno">

**Bola pesada** (Polyak, 1964): $\vw_{t+1} = \vw_t - \eta\nabla E(\vw_t) + \beta(\vw_t - \vw_{t-1})$. Lo que zigzaguea se cancela; lo que apunta siempre al mismo lado se acumula.

</div>
<div class="pequeno" v-click>

Con la **misma** $\eta$: 346 iteraciones sin momento, 79 con $\beta = 0.64$. Pero con $\beta = 0.9$ se pasa del mínimo y tarda 232: el momento también se puede exagerar.

</div>
</div>

<!--
Interactivo: empezar con β = 0, luego 0.5, 0.64 y 0.9. Es el valle de la Unidad 4 (temperatura sin estandarizar): κ = λmax/λmin grande obliga a una tasa chica.
-->

---

# Por qué funciona: una recurrencia de segundo orden

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

En $E = \frac12\vw\T H\vw$, en la dirección de cada valor propio $\lambda$:

$$
e_{t+1} = (1 + \beta - \eta\lambda)\,e_t - \beta\, e_{t-1}
$$

$$
r^2 - (1+\beta-\eta\lambda)\,r + \beta = 0 .
$$

<v-click>

Si las raíces son complejas, $|r|^2 = \beta$: **la misma tasa $\sqrt\beta$ en todas las direcciones**. Eso pasa si $(1-\sqrt\beta)^2 \le \eta\lambda \le (1+\sqrt\beta)^2$; pidiendo que $[\lambda_{\min}, \lambda_{\max}]$ quepa:

$$
\sqrt\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} \quad \text{contra} \quad \frac{\kappa - 1}{\kappa + 1} \text{ sin momento.}
$$

</v-click>
</div>
<div>

<ErrorMomento />

<div class="pequeno tenue" style="margin-top: -6px">

Punteada: $\eta$ y $\beta$ óptimos, 60 iteraciones.

</div>

<div class="pequeno" v-click>

**Estabilidad:** converge si $0 < \eta\lambda < 2(1+\beta)$. Con el mismo paso efectivo $\eta/(1-\beta)$, el momento tolera curvaturas **19 veces** mayores ($\beta = 0.9$).

</div>
</div>
</div>

<!--
Con κ = 50: 0.961 contra 0.752 por paso; iteraciones del orden de κ contra √κ. Los óptimos (η = 0.061, β = 0.57) llegan en 60 iteraciones. La condición de estabilidad sale de pedir que las dos raíces tengan módulo menor que 1 (criterio de Jury).
-->

---

# Adam: una escala por parámetro

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

$$
\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t, \quad \mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2
$$

$$
\vw_t = \vw_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon}, \quad \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \ \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}
$$

<v-click>

**¿Por qué dividir?** Con $\mathbf{m}_0 = 0$, $\mathbf{m}_t = (1-\beta_1)\sum_{i\le t}\beta_1^{t-i}\mathbf{g}_i$, así que

$$
\E[\mathbf{m}_t] = (1 - \beta_1^t)\,\E[\mathbf{g}] .
$$

</v-click>
</div>
<div>

<v-click>

<SesgoAdam />

<div class="pequeno">

Sin corregir, con $\beta_1 = 0.9$ y $\beta_2 = 0.999$ el denominador está más sesgado que el numerador: los primeros pasos salen hasta **6.6 veces** más grandes.

</div>

</v-click>
</div>
</div>

<!--
Kingma y Ba, ICLR 2015. Valores usuales: β1 = 0.9, β2 = 0.999, ε = 1e-8, η = 1e-3. El paso es ≈ η sin importar la escala del gradiente de cada coordenada: invariante a reescalar coordenadas. Ejercicio 3 de la libreta.
-->

---

# Calendarios, y el decaimiento de pesos con Adam

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<CalendariosTasa />

<div class="pequeno">

Grande para avanzar, chica para asentarse. Coseno: $\eta_t = \frac{\eta_0}{2}\big(1 + \cos\frac{\pi t}{T}\big)$. El **calentamiento** protege a la inicialización (y a los *transformers*).

</div>
</div>
<div v-click>

Con SGD, la L2 $\frac\lambda2\norm{\vw}^2$ da $\vw \leftarrow (1 - \eta\lambda)\vw - \eta\mathbf{g}$: **decaimiento de pesos**.

Con Adam, $\lambda\vw$ pasa por la división entre $\sqrt{\hat{\mathbf{v}}}$: los pesos con gradientes grandes casi no decaen. **AdamW** (Loshchilov y Hutter, 2019) lo desacopla:

$$
\vw_t = \vw_{t-1} - \eta\Big(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\varepsilon} + \lambda\vw_{t-1}\Big) .
$$

<div class="pequeno tenue">

En PyTorch: `Adam(weight_decay=λ)` es L2; `AdamW(weight_decay=λ)` es la desacoplada.

</div>
</div>
</div>

---

# Tres optimizadores, la misma red

<CurvasOptimizadores :alto="350" />

<div class="cols" style="gap: 40px; margin-top: 4px">
<div class="pequeno" v-click>

SGD ($\eta = 0.05$) avanza **a saltos**: su pérdida llega a subir 0.5 de una época a otra. El momento con el **mismo paso efectivo** ($\eta = 0.005$, $\beta = 0.9$) avanza igual en promedio y sin sobresaltos.

</div>
<div class="pequeno" v-click>

Adam ($\eta = 10^{-3}$) es el más rápido: 0.05 en entrenamiento a la época 30. Y su pérdida de **validación** toca fondo en la época 6 y luego sube hasta 0.70.

</div>
</div>

<!--
Los saltos de SGD son consistentes con el límite de estabilidad: con η = 0.05, SGD es inestable para λ > 40; con momento (η = 0.005, β = 0.9), solo para λ > 760.
-->

---
layout: seccion
numero: 3
---

# Regularización

Que lo aprendido sirva fuera del entrenamiento.

---

# El sobreajuste que la exactitud no ve

<ParadaTemprana />

<div class="cols" style="gap: 40px; margin-top: 4px">
<div class="pequeno" v-click>

La pérdida de validación casi se duplica, pero la exactitud se queda en 0.86: la red acierta lo mismo con **confianza exagerada**. Cuando falla, asigna probabilidad casi cero a la clase correcta.

</div>
<div class="pequeno" v-click>

**Parada temprana** en la época 6: pérdida de 0.41 en vez de 0.70, pero exactitud de prueba de **0.843** contra 0.858. Hay que parar según la métrica que importa.

</div>
</div>

<!--
Para un modelo lineal, parar en el paso t equivale aproximadamente a L2 con λ ≈ 1/(ηt) (Goodfellow et al., 2016, sec. 7.8). La validación usada para elegir la época ya no es una estimación honesta: por eso se reporta en prueba.
-->

---

# *Dropout*

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 44px">
<div>

En cada paso, cada neurona oculta se apaga con probabilidad $p$ (Srivastava et al., *JMLR* 2014):

$$
\tilde{\mathbf{a}} = \frac{\mathbf{r}\odot\mathbf{a}}{1-p}, \quad r_j \sim \text{Bernoulli}(1-p), \quad \E[\tilde{\mathbf{a}}] = \mathbf{a} .
$$

Al evaluar se usa $\mathbf{a}$ tal cual: el escalamiento **invertido** ya igualó las escalas.

</div>
<div>

<v-clicks>

- **Coadaptación:** ninguna neurona puede contar con que otra esté presente; cada una tiene que servir por sí sola.
- **Ensamble:** cada máscara es una de $2^n$ subredes que comparten pesos. Evaluar sin *dropout* aproxima su promedio.
- **Verificación:** promediar 100 subredes al azar da 0.865 de exactitud de validación; la red completa, 0.867. Coinciden en el 99 % de las predicciones.

</v-clicks>

</div>
</div>

<!--
El promedio es exacto para una capa lineal y aproximado después de no linealidades. Es la misma idea de los bosques aleatorios (Unidad 5): promediar modelos reduce varianza.
-->

---

# Con *dropout*, la brecha se cierra

<CurvasDropout />

<div class="pequeno" style="margin-top: 8px" v-click>

Con Adam, la pérdida de validación termina en **0.42** en lugar de 0.70 y la parada temprana ya no cuesta exactitud (0.858 contra 0.863). SGD y momento aún no sobreajustaban: el efecto es menor, aunque las dos llegan a pérdidas de validación más bajas (0.39 y 0.38).

</div>

---

# Normalización por lotes y aumento de datos

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 44px">
<div>

**Normalización por lotes** (Ioffe y Szegedy, 2015): estandarizar cada neurona con el minilote y devolverle escala y desplazamiento aprendidos,

$$
\hat z_j = \frac{z_j - \mu_{\mathcal{B},j}}{\sqrt{\sigma^2_{\mathcal{B},j} + \varepsilon}}, \qquad \gamma_j\hat z_j + \beta_j .
$$

<div class="pequeno">

Al evaluar usa promedios móviles. Ayuda sobre todo a **optimizar**: con Adam, pérdida de entrenamiento de 0.22 en la época 5 contra 0.30, y mínimo de validación de 0.394 en la época 5. Sin más regularización, igual termina memorizando (0.64).

</div>
</div>
<div v-click>

**Aumento de datos:** fabricar ejemplos con transformaciones que no cambian la clase (reflejar una prenda, desplazarla dos píxeles, cambiar el brillo).

<div class="pequeno">

Es la forma de decirle al modelo **qué invariancias esperamos** sin tocar la arquitectura. En imágenes es de lo más eficaz; las redes convolucionales lo usan siempre.

</div>

<div class="pequeno tenue">

¿Tiene sentido reflejar verticalmente un botín? Ejercicio 5 de la libreta.

</div>
</div>
</div>

<!--
La explicación original («desplazamiento de covariables interno») se ha discutido; Santurkar et al. (NeurIPS 2018) argumentan que suaviza el paisaje de la pérdida.
-->

---

# La tabla final: exactitud en prueba

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div style="font-size: 20px">

| modelo (10,000 imágenes) | última época | parada temprana |
|---|---|---|
| lineal (*softmax*) | 0.828 | 0.830 |
| SGD · con *dropout* | 0.852 · 0.853 | 0.852 · 0.853 |
| momento · con *dropout* | 0.851 · 0.855 | 0.851 · 0.856 |
| Adam · con *dropout* | 0.858 · 0.863 | 0.843 · 0.858 |
| Adam + normalización | 0.853 | 0.856 |
| **Adam + *dropout*, 50,000 imágenes** | **0.887** | **0.887** |

</div>
<div>

<v-clicks>

- Con 10,000 imágenes, todo queda entre 0.85 y 0.86. Cambia **cómo** se llega y la calidad de las probabilidades.
- **Más datos** suben 2.4 puntos: más que cualquier regularizador.

</v-clicks>

<div class="pequeno tenue" v-click>

Y un misterio: las redes tienen más parámetros que datos y aun así generalizan. El error de prueba puede bajar, subir y **volver a bajar** al crecer el modelo: el doble descenso (Belkin et al., *PNAS* 2019).

</div>
</div>
</div>

<!--
Diferencias de milésimas no significan nada sin repetir con varias semillas.
La corrida de 50,000 imágenes usa Adam, dropout, calentamiento + coseno y 15 épocas. Su pérdida de validación es 0.30, contra 0.38–0.70 con 10,000.
-->

---
layout: caso
caso: Caso · Costo ambiental
cifra: 284 <small>t CO₂e</small>
cifraNota: estimación para entrenar un transformer grande incluyendo la búsqueda de su arquitectura (626,155 lb); unas cinco veces lo que emite un auto en toda su vida útil, con combustible.
fuente: "E. Strubell, A. Ganesh y A. McCallum, «Energy and Policy Considerations for Deep Learning in NLP», ACL 2019; D. Patterson et al., «Carbon Emissions and Large Neural Network Training», arXiv:2104.10350 (2021)."
---

# Cada barrido de hiperparámetros cuesta

Strubell y coautores estimaron la energía de entrenar modelos de lenguaje: un *transformer* grande, una vez, unas 192 lb de CO₂e; repetido miles de veces dentro de una **búsqueda de arquitectura**, 626,155 lb.

<v-click>

La cifra se volvió famosa y también se discutió: Patterson et al. (2021) recalcularon esa búsqueda con el hardware y los centros de datos que se usaron de verdad y obtuvieron unas **88 veces menos**. Estimaron unas 552 t de CO₂e para entrenar GPT-3.

</v-click>

<v-click>

La lección no depende de la cifra exacta: el costo lo deciden el tamaño del modelo, **cuántas veces se entrena** (cada curva de hoy es un entrenamiento) y de dónde sale la electricidad. Reportarlo es parte de hacer bien el trabajo.

</v-click>

<!--
Las cifras de Strubell suponen GPUs P100 y la mezcla eléctrica promedio de EE. UU. Patterson et al. son de Google, la empresa que hizo la búsqueda original: vale mencionar el posible conflicto, aunque su cálculo usa datos medidos.
-->

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 24px; line-height: 1.4">

1. El gradiente de un minilote es **insesgado** con varianza $\sigma^2/B$; la tasa debe quedar por debajo de $2/\lambda_{\max}$.
2. El **momento** cambia $\kappa$ por $\sqrt\kappa$ y sube el límite de estabilidad; **Adam** da una escala por parámetro y necesita su corrección de sesgo.
3. Vigilar la **pérdida** de validación, no solo la exactitud: el sobreajuste se ve primero en las probabilidades.
4. Parada temprana, *dropout*, decaimiento de pesos (AdamW), normalización y aumento de datos; pero **más datos** ganan.
5. Cada entrenamiento cuesta energía: hay que contarlo.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Entrenar una red con Fashion-MNIST* (Unidad 6, ejemplo 2).

**Laboratorio 6:** retropropagación desde cero y entrenamiento con PyTorch.

**Lectura:** I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning* (2016), capítulos 7 y 8; S. J. D. Prince, *Understanding Deep Learning* (2023), capítulos 6 y 9.

</div>
</div>
</div>
