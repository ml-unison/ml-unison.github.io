---
theme: none
title: "Redes neuronales: perceptrón multicapa y diferenciación automática"
unidad: Unidad 6
layout: portada
subtitulo: "Capas que aprenden sus propios atributos, y bibliotecas que calculan su gradiente"
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

# Perceptrón multicapa y diferenciación automática

---
layout: pregunta
---

# Una neurona traza una recta. ¿Con qué recta separamos el «o exclusivo»?

<v-click>

Con ninguna: en el XOR, cualquier recta deja de cada lado la mitad de cada clase. Hace falta **otra capa**. Dibujarla es fácil; lo difícil, y lo que tomó años, fue saber **entrenarla**.

</v-click>

<!--
Dibujar en el pizarrón los cuatro puntos (0,0), (1,1) azules y (0,1), (1,0) rojos. Dejar que propongan rectas. La regresión logística acierta 49.5 % en las nubes del XOR de la libreta: lo mismo que una moneda.
-->

---
layout: cita
autor: The New York Times, 8 de julio de 1958
nota: "Sobre la demostración del perceptrón de Frank Rosenblatt, financiado por la Oficina de Investigación Naval de Estados Unidos (traducción propia)."
---

La Marina reveló hoy el embrión de una computadora electrónica que, espera, podrá caminar, hablar, ver, escribir, reproducirse y ser consciente de su existencia.

<!--
El perceptrón de Rosenblatt era una sola capa de pesos ajustables, que se corregían cada vez que el modelo se equivocaba en un ejemplo. La expectativa pública iba muy por delante de lo que el modelo podía hacer. Es una buena ocasión para comentar con el grupo cómo se cubren hoy los anuncios de IA.
-->

---

# De una capa a muchas, y del gradiente a mano al automático

<LineaTiempoPerceptron />

<div class="cols" style="gap: 40px; margin-top: 8px">
<div class="pequeno">

**1969–1986.** Minsky y Papert mostraron qué no puede calcular una sola capa. La salida eran varias capas, y para entrenarlas había que calcular un gradiente con miles de parámetros: la **retropropagación**.

</div>
<div class="pequeno" v-click>

**2015–2018.** Las bibliotecas calculan el gradiente de **cualquier** programa. Desde entonces, inventar una arquitectura nueva es escribir la pasada hacia adelante.

</div>
</div>

<!--
La idea del modo inverso de la diferenciación ya estaba en la tesis de maestría de S. Linnainmaa (1970); P. Werbos la propuso para redes en su tesis doctoral (1974) y el artículo de Rumelhart, Hinton y Williams en Nature 323 (1986) la popularizó. Theano (2007) fue la primera biblioteca muy usada que derivaba sola en GPU; TensorFlow (2015), PyTorch (2016) y JAX (2018) son las de hoy.
-->

---

# Dos capas doblan el plano

<div class="cols" style="grid-template-columns: 760px 1fr; gap: 36px">
<div>

<XorCapaOculta v-if="$clicks < 1" :oculto="false" />
<XorCapaOculta v-else />

</div>
<div>

Red 2-2-2 con $\tanh$ en 200 puntos del XOR. Las neuronas ocultas $a^{[1]}_1, a^{[1]}_2$ solo miran la suma $s = x_1 + x_2$ de los atributos:

$$
\begin{aligned}
a^{[1]}_1 &\approx \tanh(2.7 - 2.2\,s)\\
a^{[1]}_2 &\approx \tanh(-2.8 - 2.7\,s)
\end{aligned}
$$

<v-click>

Cada clase va a parar a su propia región del espacio oculto; ahí una recta, la logística de salida, separa las clases. Exactitud: 0.99.

</v-click>
</div>
</div>

<!--
Primero solo el espacio original: la frontera de la red ya no es una recta. Con el clic aparece el espacio oculto: la nube azul (−1,−1) va a la esquina (1,1), la azul (1,1) a (−0.9,−1) y las dos rojas a (1,−1). Es la misma idea que la solución a mano, que también solo mira x1 + x2: a1 = ReLU(x1 + x2), a2 = ReLU(x1 + x2 − 1), ŷ = a1 − 2 a2 en las cuatro esquinas de {0,1}².
Detalle para comentar: de 20 inicializaciones al azar, 15 resuelven el XOR y 5 se quedan en un mínimo local (exactitud 0.64–0.74). La red puede, pero el descenso no siempre encuentra cómo.
-->

---
layout: seccion
numero: 1
---

# Una red aprende sus propios atributos

Las capas ocultas transforman el espacio; la última es un modelo lineal generalizado.

---

# La red en dos partes

<div class="cols" style="grid-template-columns: 6fr 5fr; gap: 56px">
<div>

Para un ejemplo $\vx^{(i)} \in \R^n$ con $n$ atributos y $K$ clases:

$$
\underbrace{\va^{[1]} = g\big(\vW^{[1]}\vx^{(i)} + \mathbf b^{[1]}\big)}_{\text{atributos aprendidos } \boldsymbol\phi_\theta(\vx^{(i)})}
\qquad
\underbrace{\hat{\mathbf y} = \operatorname{softmax}\big(\vW^{[2]}\va^{[1]} + \mathbf b^{[2]}\big)}_{\text{regresión softmax sobre } \va^{[1]}}
$$

<div class="pequeno tenue">

$\vW^{[l]}$, $\mathbf b^{[l]}$: pesos y sesgos de la capa $l$; $g$: activación, entrada por entrada; $\theta$: los parámetros de las capas ocultas.

</div>
</div>
<div>

<v-click>

Las capas ocultas hacen **ingeniería de características**, como cuando inventábamos cuadrados, productos o un kernel, pero con parámetros que **se aprenden**. Con más capas, una cadena de espacios: $\vx \to \va^{[1]} \to \va^{[2]} \to \cdots$

</v-click>

<v-click>

<div class="nota">

La última capa es un **modelo lineal generalizado** (GLM) que no sabe de dónde salieron sus atributos.

</div>

</v-click>
</div>
</div>

---

# Lo que hace la capa oculta con los dígitos

<div class="cols" style="grid-template-columns: 6fr 4fr; gap: 40px">
<div>

<RepresentacionAprendida />

</div>
<div>

Los 450 dígitos de prueba proyectados en sus dos **componentes principales**, antes y después de la capa oculta de una red 64-64-10. Cada número marca el centro de su clase.

<v-click>

La **silueta** mide qué tan juntos quedan los de una clase y qué tan lejos de las demás: se **duplica**.

</v-click>

<v-click>

<div class="pequeno tenue">

La red estiró las direcciones que distinguen clases: justo lo que necesita la regresión *softmax* de la última capa.

</div>

</v-click>
</div>
</div>

<!--
En el espacio aprendido el 0, el 4 y el 6 se apartan y el 1 deja de estar encimado con el 7 y el 8. La silueta (0.17 → 0.34) se calcula en las 64 dimensiones completas, con las clases verdaderas como grupos.
-->

---

# La última capa es una logística

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 56px">
<div>

**Experimento.** Entrenamos la red. Luego tiramos su última capa, calculamos los atributos $\va^{[1]}$ de cada ejemplo y le damos esos números a una `LogisticRegression` de scikit-learn, como si alguien los hubiera diseñado.

<v-click>

| lunas, 2,000 puntos nuevos | exactitud |
|---|---|
| logística sobre $\vx$ (2 atributos) | 0.857 |
| logística sobre $\va^{[1]}$ (16 atributos) | **0.941** |
| red 2-16-2 completa | 0.946 |

</v-click>
</div>
<div>

<v-click>

| dígitos, 450 de prueba | exactitud |
|---|---|
| logística sobre los píxeles | 0.971 |
| logística sobre $\va^{[1]}$ (64 atributos) | 0.971 |
| red 64-64-10 completa | 0.973 |

</v-click>

<v-click>

<div class="nota">

Todo lo que la red gana sobre la logística está en la **representación**. Los dígitos de $8 \times 8$ ya eran casi lineales: ahí no hay mucho que ganar.

</div>

</v-click>
</div>
</div>

<!--
En las lunas el salto es de 0.857 a 0.941 solo por cambiar de espacio. En los dígitos, con 450 imágenes cada error vale 0.2 puntos: 12 contra 13 errores es ruido.
-->

---

# Por eso usamos las mismas pérdidas

La pérdida es menos el logaritmo de la verosimilitud del GLM de la última capa, con su **liga canónica** $h$: $\hat y = h(z)$, con $z$ la entrada lineal de la última capa.

| tarea | salida $\hat y = h(z)$ | modelo de $y$ | pérdida $\ell$ por ejemplo | en PyTorch |
|---|---|---|---|---|
| regresión | $z$ | normal | $\tfrac12(\hat y - y)^2$ | `MSELoss` |
| dos clases | $\sigma(z)$ | Bernoulli | $-y\ln\hat y - (1-y)\ln(1-\hat y)$ | `BCEWithLogitsLoss` |
| $K$ clases | $\operatorname{softmax}(\vz)$ | categórica | $-\ln \hat y_{y}$ | `CrossEntropyLoss` |
| conteos | $e^{z}$ | Poisson | $\hat y - y\ln \hat y$ | `PoissonNLLLoss` |

<v-click>

<div class="cols" style="gap: 48px; margin-top: 18px">
<div>

En los cuatro casos, $\;\partial \ell / \partial z = \hat y - y$: **predicción menos dato**, como en la regresión logística. En la libreta lo comprueba `autograd`.

</div>
<div class="pequeno tenue">

Las funciones de PyTorch reciben $z$ (los *logits*), no $\hat y$: aplican $h$ por dentro de forma numéricamente estable. `MSELoss` no lleva el $\tfrac12$, así que su derivada es $2(\hat y - y)$.

</div>
</div>

</v-click>

---

# Después, la pérdida cambia

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 56px">
<div>

En el perceptrón multicapa y en buena parte de las redes que usaremos, la regla es la de los GLM: **atributos aprendidos + verosimilitud**.

<v-click>

<div class="nota">

Lo que **no** cambia: la red es un programa que termina en un número, la pérdida, y su gradiente se obtiene con una llamada a `backward()`.

</div>

</v-click>
</div>
<div>

<v-click>

En redes más sofisticadas la pérdida ya no es la de un GLM:

- **contrastiva**: acercar en el espacio aprendido dos versiones del mismo ejemplo y alejar las de ejemplos distintos;
- **adversarial**: una red genera ejemplos y otra intenta distinguirlos de los reales;
- **eliminar ruido**: aprender a quitarle ruido a una imagen, para después generarlas.

</v-click>
</div>
</div>

<!--
No profundizar: se trata de que vean que la pérdida es una decisión de diseño y que la herramienta (el gradiente automático) es la misma.
-->

---

# Sin no linealidad no hay red

<div class="cols" style="grid-template-columns: 2fr 3fr; gap: 40px">
<div>

Si la activación es la identidad, $g(z) = z$, la composición de capas es **afín**:

$$
\vW^{[2]}\big(\vW^{[1]}\vx + \mathbf b^{[1]}\big) + \mathbf b^{[2]} = \vW\vx + \mathbf b ,
$$

con $\vW^{[l]}$ y $\mathbf b^{[l]}$ los pesos y sesgos de la capa $l$.

<v-click>

La «ingeniería de características» no puede hacer nada que la última capa no hiciera ya: el resultado es una regresión *softmax* con una parametrización complicada.

</v-click>

<v-click>

<div class="pequeno">

**Activaciones** $g$: sigmoide $\sigma(z) = \frac{1}{1 + e^{-z}}$, $\tanh z$ y la más usada hoy, $\operatorname{ReLU}(z) = \max(0, z)$.

</div>

</v-click>
</div>
<div>

<LunasFrontera comparar />

<div class="pequeno tenue" style="margin-top: 10px">

400 puntos de las «lunas», 16 neuronas ocultas en las dos redes; exactitud de entrenamiento. La regresión logística acierta 0.843.

</div>
</div>
</div>

---

# ¿Qué tan expresiva es una capa oculta?

<AproxReluHermosillo />

<div class="pequeno" style="margin-top: 4px">

$\hat f(x) = c_0 + \sum_{r=1}^{H} s_r \operatorname{ReLU}(x - \tau_r)$, con $x$ el día del año y $H$ neuronas ocultas, es lineal por pedazos: parte de la altura $c_0$ y cambia de pendiente en $s_r$ en cada nodo $\tau_r$. Con una neurona por nodo **interpolamos** cualquier curva. Temperatura media diaria, aeropuerto de Hermosillo (GHCN-Daily, NOAA).

</div>

<!--
Mover el deslizador: 1, 2, 3, 6, 12, 24. Los pesos no se entrenan: se construyen. Neurona r: peso 1, sesgo −τ_r; salida: s_r = cambio de pendiente en τ_r. Con 3 neuronas, RECM 1.87 °C; con 12, 0.59 °C; con 24, 0.48 °C, y ya solo persigue el ruido del promedio. Error de la interpolación lineal ≤ Δ² max|f''| / 8: baja como 1/H².
-->

---

# El teorema de aproximación universal

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<Caja tipo="teorema" nombre="Cybenko, 1989; Hornik, 1991">

Sea la activación $g$ continua, acotada y no constante. Para toda $f$ continua en un compacto $\Omega \subset \R^n$ y todo $\varepsilon > 0$ existen $H$ neuronas ocultas, con pesos $\vw_r$, sesgos $b_r$ y pesos de salida $c_r$, tales que

$$
\Big| f(\vx) - \sum_{r=1}^{H} c_r\, g(\vw_r\T\vx + b_r) \Big| < \varepsilon \quad \text{en } \Omega .
$$

</Caja>

<div class="pequeno tenue">

Leshno et al. (1993): basta con que $g$ no sea un polinomio; incluye ReLU.

</div>
</div>
<div>

**Lo que no dice:**

<v-clicks>

- **Cuántas** neuronas: con una malla de nodos se necesitan del orden de $(1/\varepsilon)^{n/2}$, con $n$ atributos; con $n = 64$ es absurdo.
- **Cómo encontrarlas**: en el XOR, la red 2-2-2 se atora en un mínimo local en 5 de 20 inicializaciones.
- Nada de **generalización**: lo que puede aproximar cualquier función también puede memorizar el ruido.

</v-clicks>

<div v-click class="nota">

Lo que sí dice: una capa ancha no tiene un **sesgo de aproximación** irreducible, como lo tiene un modelo lineal.

</div>
</div>
</div>

---
layout: seccion
numero: 2
---

# Gradientes automáticos

Una red es un grafo de cómputo, y su gradiente se calcula solo.

---
layout: pregunta
---

# La red de los dígitos tiene 4,810 parámetros. ¿Cuánto cuesta calcular su gradiente?

<v-click>

Sin saber nada de la red: mover un parámetro $\theta_j$ un poquito $h$ hacia cada lado y ver cuánto cambia la pérdida $L$,
$$
\frac{\partial L}{\partial \theta_j} \approx \frac{L(\theta + h\,\mathbf e_j) - L(\theta - h\,\mathbf e_j)}{2h}
$$
($\mathbf e_j$: vale 1 en la posición $j$ y 0 en las demás). Son **dos evaluaciones de $L$ por parámetro**.

</v-click>

---

# El parteaguas de 1986 fue de costo

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Red 64-64-10 para los dígitos: **4,810** parámetros.

<div class="comparacion" style="margin-top: 22px">
<div>
<span class="comparacion__etiqueta">diferencias finitas</span>
<span class="comparacion__cifra tenue">9,620</span>
<span class="comparacion__nota">evaluaciones de la pérdida: dos por parámetro</span>
</div>
<div v-click>
<span class="comparacion__etiqueta">retropropagación</span>
<span class="comparacion__cifra">1 + 1</span>
<span class="comparacion__nota">una pasada hacia adelante y una hacia atrás, de costo parecido</span>
</div>
</div>

</div>
<div>

<v-click>

En la libreta: el mismo gradiente (difieren en $10^{-9}$), pero las diferencias finitas tardan **más de mil veces** más.

</v-click>

<v-click>

<div class="nota">

El costo de la retropropagación **no crece** con el número de parámetros. Sin eso no habría redes de miles de millones de parámetros.

</div>

</v-click>

<v-click>

<div class="pequeno tenue">

Las diferencias finitas se quedan para **verificar** una operación nueva en una red diminuta: `torch.autograd.gradcheck`.

</div>

</v-click>
</div>
</div>

---
clicks: 2
---

# Una red es un grafo de cómputo

<GrafoComputo :paso="$clicks" />

<div class="cols" style="gap: 40px">
<div class="pequeno">

Una neurona logística con un atributo $x = 1.5$, clase $y = 1$, pesos $w = -0.8$, $b = 0.4$: $\;u = wx$, $\;z = u + b$, $\;\hat y = \sigma(z)$, $\;\ell = -\ln \hat y$. **Hacia adelante** se evalúa cada nodo en orden.

</div>
<div class="pequeno" v-click="2">

**Hacia atrás** se calcula el **adjunto** $\partial\ell/\partial v$ de cada valor $v$, de la salida a las hojas, multiplicando por la derivada **local** de cada nodo: $\partial\ell/\partial z = -\frac{1}{\hat y}\cdot\hat y(1-\hat y) = \hat y - y$.

</div>
</div>

<!--
Primer clic: los valores hacia adelante. Segundo clic: los adjuntos en rojo. Hacer las cuentas en el pizarrón: −1/0.31 = −3.226; σ' = 0.31 · 0.69 = 0.214; −3.226 · 0.214 = −0.690. El + copia el adjunto a sus dos entradas; el × lo multiplica por la otra entrada: ∂ℓ/∂w = −0.690 · 1.5 = −1.035.
-->

---

# Cada nodo solo conoce su derivada local

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Para un nodo $v = f(u_1, u_2)$, el adjunto que llega se **reparte** entre sus entradas:

$$
\frac{\partial\ell}{\partial u_k} \mathrel{+}= \frac{\partial\ell}{\partial v}\,\frac{\partial f}{\partial u_k}
$$

<div class="pequeno tenue">

Se **suma** ($+=$) porque un valor puede alimentar a varios nodos: en $f(x) = x\cdot x + x$, la $x$ recibe tres contribuciones, $x + x + 1 = 2x + 1$.

</div>

<v-click>

El algoritmo completo: ordenar los nodos (orden topológico), poner $\partial\ell/\partial\ell = 1$ y recorrer la lista al revés. En la libreta, un motor así ocupa **40 líneas** de Python.

</v-click>
</div>
<div>

| nodo | qué le pasa al adjunto $\bar v$ |
|---|---|
| $u_1 + u_2$ | llega igual a las dos entradas |
| $u_1 \cdot u_2$ | a $u_1$ le llega $\bar v\, u_2$, y al revés |
| $\tanh u$ | se multiplica por $1 - \tanh^2 u$ |
| $\operatorname{ReLU}(u)$ | pasa si $u > 0$, se anula si no |
| $\vW\va$ | a $\va$ le llega $\vW\T\bar{\mathbf v}$; a $\vW$, $\bar{\mathbf v}\,\va\T$ |

<div class="pequeno tenue">

$\bar v = \partial\ell/\partial v$ es el adjunto de la salida del nodo.

</div>
</div>
</div>

---

# Modo directo contra modo inverso

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 40px">
<div>

También se puede derivar **hacia adelante**, propagando junto con cada valor su derivada respecto de **un** parámetro: una pasada por parámetro, como las diferencias finitas.

<v-click>

El modo **inverso** da todas las derivadas de **una** salida en una pasada, y una pérdida $L: \R^P \to \R$ tiene una sola salida.

</v-click>

<v-click>

<div class="pequeno">

En una red, el modo inverso es la **retropropagación**. Con $\boldsymbol\delta^{[l]} = \partial\ell/\partial\vz^{[l]}$ y la tabla anterior sale
$\boldsymbol\delta^{[l]} = \big(\vW^{[l+1]\top}\boldsymbol\delta^{[l+1]}\big) \odot g'\big(\vz^{[l]}\big)$: el error viaja hacia atrás por las mismas matrices, transpuestas ($\odot$: entrada por entrada).

</div>

</v-click>
</div>
<div>

<CostoModos />

<div class="pequeno tenue">

Gradiente de una pérdida logística con $M = 500$ ejemplos: `torch.func.jacfwd` (directo) contra `jacrev` (inverso). Tiempos de una laptop.

</div>
</div>
</div>

<!--
Con P = 3,000 el modo directo ya es unas 70 veces más lento, y la brecha crece linealmente con P. El modo directo sirve cuando hay pocas entradas y muchas salidas, que no es nuestro caso.
-->

---

# PyTorch arma el grafo mientras el código corre

<div class="cols codigo-chico" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

```python
w = torch.tensor(-0.8, requires_grad=True)
b = torch.tensor(0.4, requires_grad=True)
perdida = -torch.log(torch.sigmoid(w * 1.5 + b))
perdida.backward()          # recorre el grafo al revés
w.grad, b.grad              # (−1.035, −0.690)
```

<v-click>

```text
perdida.grad_fn:
NegBackward0
   LogBackward0
      SigmoidBackward0
         AddBackward0
            MulBackward0
               AccumulateGrad      ← w.grad
            AccumulateGrad         ← b.grad
```

</v-click>
</div>
<div>

<v-clicks>

- Un tensor con `requires_grad=True` es nuestro nodo: cada operación con él guarda en `grad_fn` cómo derivarla.
- El grafo es **dinámico**: se construye en cada evaluación, así que admite `if`, ciclos y funciones de Python.
- `backward()` hace el recorrido hacia atrás y deja el gradiente en el `.grad` de las hojas.
- Los mismos números que el grafo de hace dos láminas.

</v-clicks>

</div>
</div>

---

# Tres cosas de autograd que causan errores

<div class="cols codigo-chico" style="grid-template-columns: 1fr 1fr; gap: 44px">
<div>

**1. Los gradientes se acumulan.** `backward()` **suma** en `.grad`, como el $+=$ de la regla local:

```python
x = torch.tensor(3.0, requires_grad=True)
(x * x + x).backward()   # x.grad = 7
(x * x + x).backward()   # x.grad = 14 (!)
```

Por eso el ciclo de entrenamiento empieza con `optimizador.zero_grad()`.

</div>
<div>

<v-click>

**2. Actualizar no se deriva.** El paso $\theta \leftarrow \theta - \eta\,\nabla_\theta L$ ($\eta$: tasa de aprendizaje) va fuera del grafo:

```python
with torch.no_grad():
    for p in red.parameters():
        p -= eta * p.grad
```

</v-click>

<v-click>

**3. `detach()`** corta el grafo: el tensor que devuelve es un número fijo para autograd. Sirve para guardar resultados o graficar sin arrastrar el grafo.

</v-click>
</div>
</div>

---

# El mismo gradiente en tres bibliotecas

<div class="cols codigo-chico" style="grid-template-columns: 1fr 1fr 1fr; gap: 24px">
<div>

**PyTorch**

```python
w = torch.tensor(-0.8,
      requires_grad=True)
l = -torch.log(
      torch.sigmoid(w*1.5 + b))
l.backward()
w.grad
```

</div>
<div>

**TensorFlow**

```python
w = tf.Variable(-0.8)
with tf.GradientTape() as cinta:
    l = -tf.math.log(
      tf.sigmoid(w*1.5 + b))
cinta.gradient(l, w)
```

</div>
<div>

**JAX**

```python
def perdida(w, b):
    return -jnp.log(
      jax.nn.sigmoid(w*1.5 + b))

jax.grad(perdida)(-0.8, b)
```

</div>
</div>

<div class="cols" style="gap: 44px; margin-top: 14px">
<div class="pequeno">

<v-click>

Las tres graban un grafo y lo recorren en modo inverso. Las tres pueden además **compilarlo** para que corra más rápido en GPU: `torch.compile`, `tf.function`, `jax.jit`.

</v-click>

</div>
<div class="pequeno">

<v-click>

**Usamos PyTorch**: su grafo dinámico se escribe y se depura como Python normal, y es la biblioteca de la mayor parte de la investigación y de los modelos abiertos que se publican. Dominar una de ellas es lo que permite construir cualquier arquitectura.

</v-click>

</div>
</div>

<!--
Las tres dan −1.035. TensorFlow graba las operaciones en una «cinta»; JAX es funcional: transforma una función de Python en la función que calcula su gradiente. Lo que se aprende con PyTorch se traslada casi directamente.
-->

---

# Entrenar: descenso por gradiente con minilotes

<LunasFrontera>
<div class="pequeno codigo-chico" style="margin-top: 14px">

```python
for xb, yb in minilotes:
    optimizador.zero_grad()
    F.cross_entropy(red(xb), yb).backward()
    optimizador.step()        # θ ← θ − η ∇θ L
```

</div>
<div class="pequeno" v-click>

Red 2-16-2 con $\tanh$, minilotes de 20, $\eta = 0.3$. Exactitud en 2,000 puntos nuevos: **0.946**, contra 0.857 de la regresión logística.

</div>
</LunasFrontera>

<!--
Recorrer las épocas con los botones: 0, 5, 20, 100, 300. Con pesos chicos, tanh z ≈ z: en la época 0 la red es casi lineal. Después la frontera se dobla para seguir las lunas. Los picos de la curva son ruido de los minilotes.
-->

---

# Los dígitos escritos a mano

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

1,347 dígitos de 8 × 8 de entrenamiento y 450 de prueba. Red 64-64-10 con ReLU en PyTorch, minilotes de 32, $\eta = 0.1$, 40 épocas: **0.3 segundos**.

| modelo | exactitud | errores |
|---|---|---|
| logística, $C = 10$ | 0.971 | 13 |
| MLP 64-64-10 | 0.973 | 12 |

<v-click>

<div class="pequeno tenue">

En la logística, $C = 1/\lambda$ es el inverso de la regularización. Con 450 imágenes, cada error vale 0.2 puntos: **no hay una ventaja clara**. Las redes se lucen con imágenes más grandes.

</div>

</v-click>
</div>
<div>

<ErroresDigitosRed />

<div class="pequeno tenue" style="margin-top: 12px">

Los 12 errores de la red: dígito real → predicho.

</div>
</div>
</div>

<!--
Un SVM con kernel gaussiano llega a 99.3 % en estos datos: la red pequeña no lo supera aquí.
-->

---
layout: seccion
numero: 3
---

# Inicialización

Que la señal y el gradiente sobrevivan a muchas capas.

---
layout: pregunta
---

# ¿Por qué no empezar con todos los pesos en cero?

<v-click>

Porque todas las neuronas ocultas calcularían lo mismo, recibirían **el mismo gradiente** y seguirían iguales para siempre. En la libreta, con todos los pesos en 0.1, después de 40 épocas las 64 neuronas siguen siendo **una**: exactitud 0.23.

</v-click>

<v-click>

Hay que **sortear** los pesos para romper la simetría. ¿De qué tamaño?

</v-click>

---

# El tamaño de los pesos se multiplica

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 40px">
<div>

Una neurona suma $n_{\text{in}}$ entradas $a_j$ con pesos $w_j$: $z = \sum_{j=1}^{n_{\text{in}}} w_j a_j$. Con pesos independientes, de media cero:

$$
\Var(z) = n_{\text{in}}\,\Var(w)\,\E\big[a^2\big] .
$$

<v-click>

Con $n_{\text{in}} = 256$ entradas de varianza 1:

| pesos | desviación de $z$ |
|---|---|
| $w \sim N(0, 1)$ | $\sqrt{256} = 16$ |
| $w \sim N(0, 0.01^2)$ | $0.16$ |
| $w \sim N(0, 1/n_{\text{in}})$ | $1$ |

</v-click>

</div>
<div>

<EscalaInicial />

<div class="pequeno tenue">

$z$ para 10,000 ejemplos con 256 entradas normales estándar. Con 20 capas, el factor se aplica 20 veces: la señal **explota** o **se apaga**.

</div>
</div>
</div>

---

# Xavier y He: la varianza que conserva la escala

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

### Xavier o Glorot (2010) · para $\tanh$ y sigmoide

$$
\Var(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

<div class="pequeno">

Cerca de 0, $\tanh z \approx z$: hacia adelante hace falta $n_{\text{in}}\Var(w) = 1$. Hacia atrás el gradiente pasa por $\vW\T$, donde cuentan las $n_{\text{out}}$ salidas: haría falta $n_{\text{out}}\Var(w) = 1$. Se toma el **promedio**.

</div>
</div>
<div>

### He o Kaiming (2015) · para ReLU

$$
\Var(w) = \frac{2}{n_{\text{in}}}
$$

<div class="pequeno">

ReLU anula la mitad de las $z$: $\E[\operatorname{ReLU}(z)^2] = \tfrac12\Var(z)$. Para compensar esa mitad se **duplica** la varianza.

</div>
</div>
</div>

<v-click>

<div class="nota" style="margin-top: 20px">

En los dos casos el sesgo empieza en 0: la asimetría ya la ponen los pesos.

</div>

</v-click>

---

# Cómo se sortean los pesos

<div class="cols" style="grid-template-columns: 6fr 5fr; gap: 44px">
<div>

Decidida la varianza, se sortea de una **normal** $N(0, \Var(w))$ o de una **uniforme** $U(-r, r)$. Como la uniforme tiene varianza $r^2/3$, para la misma varianza $r = \sqrt{3\Var(w)}$.

<v-click>

**Ejemplo:** una capa de 64 entradas y 64 salidas.

| | $\Var(w)$ | desviación | límite $r$ |
|---|---|---|---|
| Xavier | $2/128$ | 0.125 | $\sqrt{6/128} = 0.217$ |
| He | $2/64$ | 0.177 | $\sqrt{6/64} = 0.306$ |

</v-click>
</div>
<div class="codigo-chico">

<v-click>

```python
capa = nn.Linear(64, 64)
nn.init.xavier_uniform_(capa.weight)   # tanh
nn.init.kaiming_normal_(capa.weight,   # ReLU
                        nonlinearity="relu")
nn.init.zeros_(capa.bias)
```

</v-click>

<v-click>

<div class="pequeno">

Por omisión, `nn.Linear` sortea en $U(-1/\sqrt{n_{\text{in}}}, 1/\sqrt{n_{\text{in}}})$: $\Var(w) = 1/(3n_{\text{in}})$, más chica que He. En redes poco profundas casi no importa; en redes profundas con ReLU conviene pedir He.

</div>

</v-click>
</div>
</div>

---

# La escala de los pesos, capa por capa

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 36px">
<div>

20 capas ReLU de 256 neuronas, alimentadas con los dígitos estandarizados, con cuatro formas de sortear los pesos.

<v-clicks>

- **He**: la escala se mantiene hasta la capa 20.
- **Xavier**, pensada para $\tanh$: cada capa ReLU divide la varianza entre 2; en la capa 20, la desviación es unas 500 veces menor que en la primera.
- $N(0, 0.01^2)$: factor $256 \times 10^{-4}/2 \approx 0.013$ por capa; la señal desaparece.
- $N(0, 1)$: factor $256/2 = 128$ por capa; activaciones de $10^{20}$.

</v-clicks>
</div>
<div>

<ActivacionesPorCapa />

<div class="pequeno tenue">

Desviación estándar de las activaciones en cada capa.

</div>
</div>
</div>

---

# El desvanecimiento del gradiente

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 36px">
<div>

Hacia atrás, cada capa $l$ multiplica por $\vW^{[l]\top}$ y por la derivada de la activación, $g'(\vz^{[l]})$. Con la sigmoide $\sigma(z) = 1/(1 + e^{-z})$,

$$
\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big) \le \tfrac14 .
$$

<v-click>

Red de 10 capas de 64: con sigmoides, la capa 1 recibe un gradiente **455,000 veces** menor que la capa 10, cerca de un factor de 4 por capa. Las primeras capas casi no aprenden.

</v-click>

<v-click>

<div class="nota">

Remedios: ReLU, buena inicialización, normalización por lotes, conexiones residuales y, en las redes recurrentes, las compuertas de la LSTM.

</div>

</v-click>
</div>
<div>

<GradientePorCapa />

<div class="pequeno tenue">

Norma del gradiente por capa en la inicialización, con 256 dígitos.

</div>
</div>
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 22px; line-height: 1.38">

1. Una red es **ingeniería de características aprendida** más un **GLM** en la última capa: por eso usa las mismas pérdidas y su error de salida es $\hat y - y$. Sin no linealidad, todo es un modelo lineal.
2. Una red es un **grafo de cómputo**. La diferenciación automática en modo inverso lo recorre hacia atrás con derivadas locales y da todo el gradiente al costo de unas pocas pasadas, sin importar cuántos parámetros haya. La retropropagación es ese modo inverso, y fue el parteaguas.
3. PyTorch, TensorFlow y JAX derivan cualquier programa. En PyTorch: el grafo es dinámico, los gradientes se acumulan y la actualización va en `no_grad`.
4. Los pesos se sortean para romper la simetría, con la escala que conserva la señal: **Xavier**, $2/(n_{\text{in}} + n_{\text{out}})$, para $\tanh$; **He**, $2/n_{\text{in}}$, para ReLU.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Una red neuronal con PyTorch: grafos de cómputo y diferenciación automática* (Unidad 6, ejemplo 1).

**Laboratorio 6:** diferenciación automática y entrenamiento con PyTorch.

**Lectura:** I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, secciones 6.1–6.5; A. G. Baydin et al., «Automatic differentiation in machine learning: a survey», *JMLR* 18 (2018), secciones 1–3.

</div>
</div>
</div>
