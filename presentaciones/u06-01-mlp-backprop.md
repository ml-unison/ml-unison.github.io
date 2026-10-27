---
theme: none
title: "Redes neuronales: perceptrón multicapa y retropropagación"
unidad: Unidad 6
layout: portada
subtitulo: "Capas que aprenden su propia representación, y cómo calcular su gradiente"
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
  Reconocimiento de Patrones · Unidad 6 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Perceptrón multicapa y retropropagación

---
layout: pregunta
---

# Una neurona traza una recta. ¿Con qué recta separamos el «o exclusivo»?

<v-click>

Con ninguna: en el XOR, cualquier recta deja de cada lado la mitad de cada clase. Hace falta **otra capa**. Dibujarla es fácil; lo difícil, y lo que tomó años, fue saber **entrenarla**.

</v-click>

<!--
Dibujar en el pizarrón los cuatro puntos (0,0), (1,1) azules y (0,1), (1,0) rojos. Dejar que propongan rectas. La logística de la Unidad 4 acierta 49.5 % en las nubes del XOR de la libreta: lo mismo que una moneda.
-->

---
layout: cita
autor: The New York Times, 8 de julio de 1958
nota: "Sobre la demostración del perceptrón de Frank Rosenblatt, financiado por la Oficina de Investigación Naval de Estados Unidos (traducción propia)."
---

La Marina reveló hoy el embrión de una computadora electrónica que, espera, podrá caminar, hablar, ver, escribir, reproducirse y ser consciente de su existencia.

<!--
El perceptrón de Rosenblatt era una sola capa de pesos ajustables: el mismo algoritmo del perceptrón que vimos en la Unidad 3. La expectativa pública iba muy por delante de lo que el modelo podía hacer. Es una buena ocasión para comentar con el grupo cómo se cubren hoy los anuncios de IA.
-->

---

# De una capa a muchas

<LineaTiempoPerceptron />

<div class="cols" style="gap: 40px; margin-top: 8px">
<div class="pequeno">

**1969.** En *Perceptrons*, M. Minsky y S. Papert demostraron qué no puede calcular un perceptrón de una capa; el XOR es el ejemplo más pequeño.

</div>
<div class="pequeno" v-click>

**1986.** D. Rumelhart, G. Hinton y R. Williams mostraron en *Nature* 323 que la **retropropagación** entrena redes con capas ocultas y que esas capas aprenden representaciones útiles.

</div>
</div>

<!--
La idea del modo inverso de la diferenciación ya estaba en la tesis de maestría de S. Linnainmaa (1970), y P. Werbos la propuso para redes en su tesis doctoral (1974). El artículo de 1986 la popularizó. Cybenko (1989) y Glorot y Bengio (2010) aparecen más adelante en esta clase; AlexNet, en el tema 3.
-->

---

# Dos capas doblan el plano

<div class="cols" style="grid-template-columns: 760px 1fr; gap: 36px">
<div>

<XorCapaOculta v-if="$clicks < 1" :oculto="false" />
<XorCapaOculta v-else />

</div>
<div>

Red 2-2-2 con $\tanh$, entrenada en 200 puntos del XOR. Aprende dos **franjas** paralelas a la diagonal:

$$
\begin{aligned}
a_1 &\approx \tanh\big(2(x_2 - x_1 + 1)\big)\\
a_2 &\approx \tanh\big(2(x_1 - x_2 + 1)\big)
\end{aligned}
$$

<v-click>

Las dos nubes azules van a la **misma esquina** $(1, 1)$ del espacio oculto. Ahí una recta, la logística de salida, separa las clases. Exactitud: 0.975.

</v-click>
</div>
</div>

<!--
Primero solo el espacio original: la frontera de la red ya no es una recta. Con el clic aparece el espacio oculto. La idea central de las redes: aprender una representación en la que el problema se vuelve lineal. Con pesos a mano también sale: h1 = ReLU(x1 + x2), h2 = ReLU(x1 + x2 − 1), y = h1 − 2 h2 en las cuatro esquinas de {0,1}².
-->

---

# Sin no linealidad no hay red

<div class="cols" style="grid-template-columns: 2fr 3fr; gap: 40px">
<div>

Si $\sigma(z) = z$, la composición de capas es **afín**:

$$
W^{(2)}\big(W^{(1)}\vx + \mathbf b^{(1)}\big) + \mathbf b^{(2)} = W\vx + \mathbf b .
$$

<v-click>

Con cualquier número de capas, el resultado es una regresión *softmax* con una parametrización complicada.

</v-click>

<v-click>

<div class="pequeno">

**Activaciones:** sigmoide $\frac{1}{1 + e^{-z}}$, $\tanh z$ y la más usada hoy, $\operatorname{ReLU}(z) = \max(0, z)$.

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

$g(x) = c_0 + \sum_i s_i \operatorname{ReLU}(x - t_i)$ es lineal por pedazos: con una neurona por nodo **interpolamos** cualquier curva. Temperatura media diaria en el aeropuerto de Hermosillo (GHCN-Daily, NOAA).

</div>

<!--
Mover el deslizador: 1, 2, 3, 6, 12, 24. Los pesos no se entrenan: se construyen. Neurona i: peso 1, sesgo −t_i; salida: s_i = cambio de pendiente en t_i. Con 3 neuronas, RECM 1.87 °C; con 12, 0.59 °C; con 24, 0.48 °C, y ya solo persigue el ruido del promedio. Error de la interpolación lineal ≤ Δ² max|f''| / 8: baja como 1/H².
-->

---

# El teorema de aproximación universal

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

<Caja tipo="teorema" nombre="Cybenko, 1989; Hornik, 1991">

Sea $\sigma$ continua, acotada y no constante. Para toda $f$ continua en un compacto $K \subset \R^d$ y todo $\varepsilon > 0$ existen $H$ y pesos tales que

$$
\Big| f(\vx) - \sum_{j=1}^{H} c_j\, \sigma(\vw_j\T\vx + b_j) \Big| < \varepsilon \quad \text{en } K .
$$

</Caja>

<div class="pequeno tenue">

Leshno et al. (1993): basta con que $\sigma$ no sea un polinomio; incluye ReLU.

</div>
</div>
<div>

**Lo que no dice:**

<v-clicks>

- **Cuántas** neuronas: la construcción con una malla de nodos necesita del orden de $(1/\varepsilon)^{d/2}$; con $d = 64$ es absurdo.
- **Cómo encontrarlas**: existen, pero nada garantiza que el descenso por gradiente llegue a ellas.
- Nada de **generalización**: lo que puede aproximar cualquier función también puede memorizar el ruido (Unidad 3).

</v-clicks>

<div v-click class="nota">

Lo que sí dice: una capa ancha no tiene un **sesgo de aproximación** irreducible, como lo tiene un modelo lineal.

</div>
</div>
</div>

---
layout: seccion
numero: 1
---

# Retropropagación

La regla de la cadena, organizada para no repetir cuentas.

---

# La red y su capa de salida

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Dos capas, $K$ clases:

$$
\begin{aligned}
\mathbf z^{(1)} &= W^{(1)}\vx + \mathbf b^{(1)}, & \mathbf a^{(1)} &= \sigma\big(\mathbf z^{(1)}\big),\\
\mathbf z^{(2)} &= W^{(2)}\mathbf a^{(1)} + \mathbf b^{(2)}, & \hat{\mathbf y} &= \operatorname{softmax}\big(\mathbf z^{(2)}\big),
\end{aligned}
$$

$$
E = -\textstyle\sum_k y_k \ln \hat y_k \qquad (\mathbf y \text{ en one-hot}).
$$

<div class="pequeno tenue">

La última capa es la regresión *softmax* de la Unidad 4, sobre atributos $\mathbf a^{(1)}$ **aprendidos**.

</div>
</div>
<div>

<v-click>

Definimos el **error** de cada capa: $\boldsymbol\delta^{(\ell)} = \partial E / \partial \mathbf z^{(\ell)}$.

</v-click>

<v-click>

Como $\ln \hat y_k = z_k - \ln\sum_j e^{z_j}$,

$$
\frac{\partial E}{\partial z_i} = -\sum_k y_k\big(\ind{k = i} - \hat y_i\big) = \hat y_i - y_i ,
$$

</v-click>

<v-click>

$$
\boldsymbol\delta^{(2)} = \hat{\mathbf y} - \mathbf y .
$$

«Predicción menos dato», como en la logística y en todo GLM con liga canónica.

</v-click>
</div>
</div>

---

# Hacia atrás, en forma matricial

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

**Pesos.** $W^{(\ell)}_{ij}$ solo afecta a $z^{(\ell)}_i$, con $\partial z^{(\ell)}_i / \partial W^{(\ell)}_{ij} = a^{(\ell-1)}_j$:

$$
\frac{\partial E}{\partial W^{(\ell)}} = \boldsymbol\delta^{(\ell)}\, \mathbf a^{(\ell-1)\top}, \qquad \frac{\partial E}{\partial \mathbf b^{(\ell)}} = \boldsymbol\delta^{(\ell)} .
$$

<div class="pequeno tenue">

Error que llega por activación que entra: un producto exterior.

</div>

<v-click>

**Errores.** $a^{(\ell)}_j$ afecta a **todas** las $z^{(\ell+1)}_i$; se suman y se pasa por $\sigma$:

$$
\boldsymbol\delta^{(\ell)} = \Big(W^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}\Big) \odot \sigma'\big(\mathbf z^{(\ell)}\big) .
$$

</v-click>
</div>
<div>

<v-click>

**Con un lote de $N$ ejemplos** (renglones), los productos exteriores sumados son productos de matrices:

```python
D2 = P.copy()
D2[np.arange(N), y] -= 1          # Δ(2) = Ŷ − Y
D2 /= N                           # E es un promedio
gW2, gb2 = D2.T @ A1, D2.sum(0)
D1 = (D2 @ W2) * dsigma(Z1)       # Δ(1)
gW1, gb1 = D1.T @ X, D1.sum(0)
```

</v-click>

<v-click>

<div class="nota">

El error viaja hacia atrás por las **mismas matrices** que la señal hacia adelante, transpuestas.

</div>

</v-click>
</div>
</div>

<!--
Hacer en el pizarrón el caso de un solo peso de la primera capa con la regla de la cadena, y luego reconocer el patrón matricial. En el código cada renglón es un ejemplo, por eso aparece D2.T @ A1 en lugar de δ aᵀ.
-->

---

# ¿Está bien? Verificación con diferencias finitas

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

Sin saber nada de la red:

$$
\frac{\partial E}{\partial \theta} \approx \frac{E(\theta + h) - E(\theta - h)}{2h}
$$

con error de truncamiento $O(h^2)$; la diferencia hacia adelante tiene $O(h)$.

<v-click>

Red 2-8-2 en las lunas, 42 parámetros, $h = 10^{-5}$: error relativo máximo entre $5 \times 10^{-11}$ y $2 \times 10^{-8}$ por grupo de parámetros. **Correcto.**

</v-click>

<v-click>

<div class="pequeno tenue">

Regla práctica: $\lesssim 10^{-7}$ está bien; $\gtrsim 10^{-2}$ es un error en el código. Un gradiente equivocado **entrena mal, pero entrena**: el error puede pasar inadvertido.

</div>

</v-click>
</div>
<div>

<DiferenciasFinitasV />

<div class="pequeno tenue">

Derivada respecto de un solo peso. Mejor error: $5 \times 10^{-8}$ hacia adelante; $6 \times 10^{-11}$ centrada.

</div>
</div>
</div>

<!--
La V: a la izquierda domina el truncamiento (pendiente 1 o 2 en escala log-log); a la derecha, el redondeo, que crece como ε/h con ε ≈ 10⁻¹⁶. El mejor h es ≈ √ε para la diferencia hacia adelante y ≈ ε^{1/3} para la centrada.
-->

---

# Por qué no entrenamos con diferencias finitas

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

Red 64-64-10 para los dígitos: **4,810** parámetros.

<div class="comparacion" style="margin-top: 22px">
<div>
<span class="comparacion__etiqueta">diferencias finitas</span>
<span class="comparacion__cifra tenue">9,620</span>
<span class="comparacion__nota">pasadas hacia adelante: dos por parámetro</span>
</div>
<div v-click>
<span class="comparacion__etiqueta">retropropagación</span>
<span class="comparacion__cifra">1 + 1</span>
<span class="comparacion__nota">una hacia adelante y una hacia atrás, del mismo costo</span>
</div>
</div>

</div>
<div>

<v-click>

**Modo inverso.** Para $E: \R^P \to \R$ (muchos parámetros, una pérdida), el gradiente completo cuesta un múltiplo pequeño y **constante** de evaluar $E$, sin importar $P$.

</v-click>

<v-click>

**Modo directo.** Propagar derivadas hacia adelante junto con los valores necesita una pasada por parámetro, como las diferencias finitas.

</v-click>

<v-click>

<div class="nota">

En la libreta: la retropropagación tarda menos de un milisegundo; las diferencias finitas, unos segundos (miles de veces más). Una red moderna tiene miles de millones de parámetros.

</div>

</v-click>
</div>
</div>

---

# PyTorch: el mismo gradiente, automático

<div class="cols codigo-chico" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

```python
pt = {k: torch.tensor(M, dtype=torch.float64, requires_grad=True)
      for k, M in p0.items()}          # los mismos pesos que en NumPy
logits = torch.tanh(X @ pt["W1"].T + pt["b1"]) @ pt["W2"].T + pt["b2"]
perdida = F.cross_entropy(logits, y)  # softmax + entropía cruzada
perdida.backward()                    # retropropagación
pt["W1"].grad                         # ∂E/∂W(1)
```

<v-click>

<div class="pequeno" style="margin-top: 18px">

Diferencia máxima con nuestro `atras`: $4 \times 10^{-17}$, el último dígito de la doble precisión.

</div>

</v-click>
</div>
<div>

<div class="pequeno">

<v-clicks>

- Cada operación con un tensor que tiene `requires_grad` se anota en un **grafo de cómputo**.
- El grafo se arma **mientras el código corre**: admite `if` y ciclos de Python.
- `backward()` lo recorre hacia atrás con la regla de la cadena: **diferenciación automática** en modo inverso.
- `F.cross_entropy` recibe **logits**: combina *softmax* y logaritmo de forma estable.

</v-clicks>

</div>
</div>
</div>

---

# Entrenar: descenso por gradiente con minilotes

<LunasFrontera>
<div class="pequeno" style="margin-top: 14px">

Red 2-16-2 con $\tanh$, minilotes de 20, $\eta = 0.3$. Con pesos chicos, $\tanh z \approx z$: en la época 0 la red es casi lineal.

</div>
<div class="pequeno" v-click>

Exactitud en 2,000 puntos nuevos: **0.943**, contra 0.857 de la regresión logística. Los picos de la curva son ruido de los minilotes (siguiente clase).

</div>
</LunasFrontera>

<!--
Recorrer las épocas con los botones: 0, 5, 20, 100, 300. Hacia la época 20 la red apenas supera a la logística; después la frontera se dobla para seguir las lunas.
-->

---

# Los dígitos de la Unidad 4

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 48px">
<div>

1,347 dígitos de entrenamiento y 450 de prueba (la misma partición). Red 64-64-10 con ReLU, minilotes de 32, $\eta = 0.1$, 40 épocas.

| modelo | exactitud | errores |
|---|---|---|
| logística, $C = 10$ | 0.971 | 13 |
| MLP en NumPy | 0.978 | 10 |
| MLP en PyTorch | 0.973 | 12 |

<v-click>

<div class="pequeno tenue">

Con 450 imágenes, cada error vale 0.2 puntos: **no hay una ventaja clara**. Los dígitos de 8 × 8 son casi lineales; las redes se lucen con imágenes más grandes (próximas clases).

</div>

</v-click>
</div>
<div>

<ErroresDigitosRed />

<div class="pequeno tenue" style="margin-top: 12px">

Los 12 errores de la red de PyTorch: dígito real → predicho.

</div>
</div>
</div>

<!--
Las dos redes no son idénticas: PyTorch inicializa con otra distribución (uniforme, parecida a Xavier), baraja con otro generador y trabaja en precisión simple. Con otra semilla cambian uno o dos errores. El SVM gaussiano de la Unidad 4 llegó a 99.3 %: la red pequeña no lo supera aquí.
-->

---
layout: seccion
numero: 2
---

# Inicialización

Que la señal y el gradiente sobrevivan a muchas capas.

---

# La escala de los pesos se multiplica capa por capa

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 36px">
<div>

Con $z = \sum_{j=1}^{n_{\text{in}}} w_j a_j$, pesos independientes de media cero:

$$
\Var(z) = n_{\text{in}} \Var(w)\, \E[a^2] .
$$

<v-click>

Si $n_{\text{in}}\Var(w) \ne 1$ (en la escala que toca), la señal **crece o se apaga exponencialmente** con la profundidad.

</v-click>

<v-clicks>

- $\tanh$: $\E[a^2] \approx \Var(z)$; promediando adelante y atrás, **Xavier**: $\Var(w) = 2/(n_{\text{in}} + n_{\text{out}})$.
- ReLU anula la mitad: $\E[a^2] = \tfrac12\Var(z)$, así que **He**: $\Var(w) = 2/n_{\text{in}}$.

</v-clicks>
</div>
<div>

<ActivacionesPorCapa />

<div class="pequeno tenue">

20 capas ReLU de 256 neuronas, alimentadas con los dígitos estandarizados.

</div>
</div>
</div>

<!--
Con He la desviación estándar se mantiene en 0.72 hasta la capa 20. Con Xavier y ReLU cada capa divide la varianza entre 2: en la capa 20 es unas 850 veces menor que en la primera. N(0, 0.01²) multiplica la varianza por 256 × 10⁻⁴ / 2 ≈ 0.013 en cada capa; N(0, 1), por 128: activaciones de 10²⁰.
-->

---

# El desvanecimiento del gradiente

<div class="cols" style="grid-template-columns: 5fr 6fr; gap: 36px">
<div>

Hacia atrás, cada capa multiplica por $W^{(\ell)\top}$ y por $\sigma'(\mathbf z^{(\ell)})$. Con la sigmoide,

$$
\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big) \le \tfrac14 .
$$

<v-click>

Red de 10 capas de 64: con sigmoides, la capa 1 recibe un gradiente **455,000 veces** menor que la capa 10, cerca de un factor de 4 por capa. Las primeras capas casi no aprenden.

</v-click>

<v-click>

<div class="nota">

Remedios del resto de la unidad: ReLU, buena inicialización, normalización por lotes, conexiones residuales y, en las redes recurrentes, las compuertas de la LSTM.

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
<div style="font-size: 25px; line-height: 1.4">

1. Una capa oculta **aprende una representación** en la que el problema es lineal; sin no linealidad, todo se colapsa a un modelo lineal.
2. Una capa ancha aproxima cualquier función continua, pero el teorema no dice cuántas neuronas, cómo encontrarlas ni si generalizan.
3. Retropropagación: el error $\boldsymbol\delta^{(\ell)}$ viaja hacia atrás por $W\T$ y el gradiente de cada capa es $\boldsymbol\delta^{(\ell)}\mathbf a^{(\ell-1)\top}$; cuesta lo que unas pocas pasadas hacia adelante.
4. Todo gradiente hecho a mano se **verifica** con diferencias centradas.
5. La escala inicial de los pesos decide si la señal sobrevive: Xavier para $\tanh$, He para ReLU.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Una red neuronal desde cero, y luego con PyTorch* (Unidad 6, ejemplo 1).

**Laboratorio 6:** retropropagación desde cero y entrenamiento con PyTorch.

**Lectura:** I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, secciones 6.1–6.5; *Learning From Data*, capítulo electrónico 7 (*Neural Networks*), secciones 7.1–7.2.

</div>
</div>
</div>
