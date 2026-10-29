# Notación y convenciones de las presentaciones

Todas las presentaciones del curso siguen estas convenciones. Las macros de KaTeX están en `setup/katex.ts`, y `node herramientas/revisar_katex.mjs [filtro]` revisa que todas las fórmulas compilen.

## Datos

| Símbolo | Significado | Macro |
|---|---|---|
| $\mathcal{D}$, $\mathcal{D}_{\text{ent}}$, $\mathcal{D}_{\text{prueba}}$ | conjunto de datos (con subíndice si hace falta) | `\Dat` |
| $M$ | número de datos de un conjunto; siempre $i = 1, \dots, M$ | |
| $n$ | número de atributos; siempre $j = 1, \dots, n$ | |
| $\mathbf{x}^{(i)} \in \mathbb{R}^n$ | instancia $i$ (vector, en negritas; superíndice con paréntesis para no confundir con una potencia) | `\vx^{(i)}` |
| $x^{(i)}_j$ | valor del atributo $j$ en la instancia $i$ (escalar) | |
| $y^{(i)}$, $\hat y^{(i)}$ | salida real y predicha de la instancia $i$ | |
| $\mathbf{X} \in \mathbb{R}^{M \times n}$, $\mathbf{y} \in \mathbb{R}^M$ | matriz de datos (una instancia por renglón) y vector de salidas | `\vX`, `\vy` |

Los vectores van en minúscula negrita, las matrices en mayúscula negrita y los escalares en letra normal.

## Parámetros y modelos

| Símbolo | Significado |
|---|---|
| $\theta$ ($\boldsymbol\theta$ si es vector) | parámetros en general (macro `\vth`) |
| $\mathbf{w}$, $w_j$ | pesos (`\vw`) |
| $b$ | sesgo (nunca $w_0$ ni el truco $x_0 = 1$) |
| $h \in \mathcal{H}$ | hipótesis y clase de hipótesis (`\Hyp`) |
| $\ell(y, \hat y)$ | pérdida de un dato |
| $L(\theta) = \frac{1}{M}\sum_{i=1}^{M} \ell\bigl(y^{(i)}, \hat y^{(i)}\bigr)$ | pérdida promedio que se minimiza |
| $\eta$ | tasa de aprendizaje |
| $\lambda$ | fuerza de la regularización |
| $\sigma(z) = 1/(1 + e^{-z})$ | función logística |
| $\Prob(\cdot)$, $\E[\cdot]$ | probabilidad y esperanza (`\Prob`, `\E`) |
| $K$, $k$ | número de clases o de grupos, e índice de clase o de grupo |
| $B$ | número de árboles o de rondas de un ensamble |

## Redes neuronales

La capa va entre corchetes: $\mathbf{W}^{[\ell]}$, $\mathbf{b}^{[\ell]}$, $\mathbf{z}^{[\ell]} = \mathbf{W}^{[\ell]}\mathbf{a}^{[\ell-1]} + \mathbf{b}^{[\ell]}$, $\mathbf{a}^{[\ell]} = g(\mathbf{z}^{[\ell]})$, con $\mathbf{a}^{[0]} = \mathbf{x}^{(i)}$. Si hace falta la instancia: $\mathbf{a}^{[\ell](i)}$.

## Series de tiempo

La serie es $y_1, \dots, y_T$, con subíndice $t$ y $T$ observaciones. El pronóstico a $h$ pasos es $\hat y_{T+h \mid T}$. Cuando la serie se convierte en una tabla de rezagos para aprender, se vuelve a $\mathbf{x}^{(i)}$, con $i = 1, \dots, M$.

## Reglas de redacción

- Toda función o variable que aparece en una transparencia se explica en esa misma transparencia (en el texto, en una nota al pie de la transparencia o junto a la fórmula).
- Cada presentación es autocontenida: no cita otras unidades ni otras presentaciones («como vimos en la Unidad 3»). Si hace falta una idea previa, se explica en una o dos líneas. Sí puede citar las libretas y los laboratorios de su misma unidad.

## Casos particulares ya decididos

- Número de hipótesis en la cota de la unión: $|\mathcal{H}|$ (no $M$).
- Tamaño del conjunto de validación: $M_{\text{val}}$; número de pliegues de la validación cruzada: $K$, con índice $k$, en las presentaciones donde no hay clases.
- Tamaño del minilote: $|\mathcal{B}|$.
- Bosques aleatorios: $n_s$ atributos sorteados en cada corte. Ensambles: $B$ árboles o rondas, con índice $t = 1, \dots, B$ (no $b$, que es el sesgo).
- Impureza de un nodo: $Q(\cdot)$.
- PCA: $r$ componentes; t-SNE: puntos proyectados $\mathbf{z}^{(i)}$.
- Log-verosimilitud: $\log p(\mathcal{D} \mid \theta)$ o $\ln \mathcal{L}$ (no $\ell$, que es la pérdida).
- Kernel: $K(\mathbf{x}, \mathbf{x}')$ en la presentación de SVM, que no usa $K$ para clases. Atención: $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$.
- Activación genérica de una red: $g$; $\sigma$ solo es la logística.
- ARIMA: $\phi_j$, $\theta_j$, el operador de retraso $B$ y el periodo estacional $m$ se conservan (notación de Box–Jenkins y Hyndman), siempre explicados en la lámina.
- Dos muestras que se comparan (deriva): $M_1$ y $M_2$, o $M_{\text{ref}}$ y $M_{\text{act}}$.
- Valores propios: $\lambda$ en regresión lineal y LDA (donde no hay regularización en la misma presentación); $\mu$ en optimización.
