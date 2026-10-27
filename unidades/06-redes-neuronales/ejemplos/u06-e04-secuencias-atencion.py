# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <small>Reconocimiento de Patrones · Licenciatura en Ciencias de la Computación · Universidad de Sonora<br>
# Julio Waissman · <a href="https://ml-unison.github.io">ml-unison.github.io</a> · Licencia CC BY-SA 4.0</small>
#
# ---
#
# # Un modelo de lenguaje del *Quijote*, letra por letra
#
# **Unidad 6 · Ejemplo 4**
#
# Si leemos «En un lugar de la Man», casi cualquiera completa la siguiente letra sin dudar. Si leemos «dijo », las opciones son muchas más. Un **modelo de lenguaje** es exactamente eso: una distribución de probabilidad para lo que sigue, dado lo que ya se leyó. Es la tarea con la que se entrenan los grandes modelos de lenguaje actuales, y aquí la vamos a resolver en pequeño, con la novela de Cervantes y unos cuantos minutos de CPU. En esta libreta vamos a:
#
# 1. plantear el problema de **predecir el siguiente carácter** y medir los modelos con la **perplejidad**;
# 2. construir la línea base clásica: **$n$-gramas** con suavizado de Laplace, y ver por qué no basta con contar contextos más largos;
# 3. entender las **redes recurrentes**, por qué su gradiente se desvanece o explota (un producto de jacobianas) y cómo lo resuelven el **recorte de gradiente** y la **LSTM**;
# 4. derivar la **atención** de producto punto escalado, incluida la razón del $\sqrt{d_k}$, y programar un ***transformer*** mínimo;
# 5. comparar los tres modelos, **generar texto** con temperatura y mirar **qué letras atiende** una cabeza de atención.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado.

# %%
import os, sys

RAMA = "rediseno-2027"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TINTA, TENUE = rdp.estilo.TINTA, rdp.estilo.TENUE

# La verificación automática del repositorio usa menos pasos de entrenamiento (RDP_RAPIDO=1);
# las cifras del texto son las de la ejecución normal (unos 3 minutos en CPU, segundos con GPU).
RAPIDO = os.environ.get("RDP_RAPIDO") == "1"
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEMILLA)
print("dispositivo:", dispositivo, "| modo rápido:", RAPIDO)

# %% [markdown]
# ## 1. Los datos y el problema
#
# **Los datos.** El texto completo de *El ingenioso hidalgo don Quijote de la Mancha* (Miguel de Cervantes; primera parte de 1605, segunda de 1615), en la edición del Proyecto Gutenberg (libro 2000, dominio público), sin el encabezado ni el pie de Gutenberg. Son unos 2.1 millones de caracteres: letras, acentos, signos de puntuación, rayas de diálogo y saltos de línea, tal cual.
#
# **El problema.** Dado un fragmento del texto, predecir **el siguiente carácter**. Más precisamente: estimar la distribución $P(x_t \mid x_1, \dots, x_{t-1})$ sobre los caracteres posibles. No hay etiquetas que alguien haya puesto a mano: la «etiqueta» de cada posición es la letra que de verdad sigue. Por eso a esta tarea se le llama a veces **autosupervisada**, y por eso se puede entrenar con todo el texto que uno tenga.
#
# **Partición.** El primer 90 % de las letras se usa para entrenar y el último 10 % para validar. No se puede partir al azar, letra por letra: cada letra de validación tendría a sus vecinas en el entrenamiento y estaríamos midiendo memoria, no predicción. El corte cae, por casualidad, justo al inicio del capítulo LIX de la segunda parte, así que validamos con los últimos 16 capítulos de la novela.

# %%
texto = rdp.datos.texto("quijote")
vocabulario = sorted(set(texto))
V = len(vocabulario)
a_indice = {c: i for i, c in enumerate(vocabulario)}
ids = np.array([a_indice[c] for c in texto], dtype=np.int64)

corte = int(0.9 * len(ids))
ids_tr, ids_va = ids[:corte], ids[corte:]
print(f"{len(texto):,} caracteres; {V} distintos")
print(f"entrenamiento: {len(ids_tr):,}   validación: {len(ids_va):,}")
print("\nInicio de la validación:\n" + texto[corte:corte + 250])

# %% [markdown]
# **Qué letras se califican.** Todos los modelos de la libreta se evalúan **en las mismas letras de validación**: las que tienen al menos 64 caracteres de contexto dentro de la parte de validación. Es casi todo el 10 %; la razón de exigir ese contexto se verá en la sección 5.

# %%
T = 128          # longitud de contexto de las redes
CONTEXTO_MIN = 64
# Ventanas de validación de T + 1 caracteres que avanzan de 64 en 64: en cada una se califican
# solo las últimas 64 predicciones, que tienen entre 65 y 128 caracteres de contexto.
inicios_va = np.arange(0, len(ids_va) - T, CONTEXTO_MIN)
objetivos_va = (inicios_va[:, None] + np.arange(CONTEXTO_MIN + 1, T + 1)).ravel()   # posiciones calificadas
print(f"{len(objetivos_va):,} letras de validación calificadas ({len(inicios_va):,} ventanas)")

# %% [markdown]
# Los 92 símbolos son minúsculas y mayúsculas (con y sin acento, y la ñ), dígitos (casi no hay), signos de puntuación, la raya de diálogo «—», el espacio y el salto de línea. Las frecuencias son muy desiguales: el espacio (que en las gráficas dibujamos «·»; el salto de línea, «¶») y la «e» son, por mucho, los más comunes.

# %%
def visible(c):
    """Espacio y salto de línea en forma visible."""
    return {" ": "·", "\n": "¶"}.get(c, c)


frecuencias = pd.Series(np.bincount(ids_tr, minlength=V), index=vocabulario).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 3.8))
top = frecuencias.iloc[:40] / frecuencias.sum()
ax.bar(range(len(top)), top.values, color=AZUL)
ax.set_xticks(range(len(top)), [visible(c) for c in top.index], fontsize=12)
ax.set_ylabel("frecuencia relativa")
ax.set_title("Los 40 caracteres más frecuentes del Quijote (entrenamiento)")
ax.grid(axis="x", visible=False)
plt.show()

# %% [markdown]
# ## 2. Modelos de lenguaje y perplejidad
#
# Por la regla de la cadena de la probabilidad, cualquier distribución sobre secuencias se puede factorizar como
#
# $$P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1}) = \prod_{t=1}^{T} P(x_t \mid x_{<t}).$$
#
# No es una suposición, es una identidad. Un **modelo de lenguaje autorregresivo** aprende cada factor: un clasificador de $V = 92$ clases cuya entrada es el pasado. Se entrena, como cualquier clasificador probabilístico de las unidades 3 y 4, minimizando la **entropía cruzada** promedio por carácter
#
# $$H = -\frac{1}{N}\sum_{t=1}^{N} \ln P_\theta(x_t \mid x_{<t}),$$
#
# es decir, maximizando la verosimilitud del texto.
#
# ### Perplejidad: el número efectivo de opciones
#
# La entropía cruzada en *nats* es difícil de imaginar. Su exponencial sí se interpreta bien:
#
# $$\text{perplejidad} = e^{H} = \Big(\prod_{t=1}^{N} \frac{1}{P_\theta(x_t \mid x_{<t})}\Big)^{1/N}.$$
#
# Es la media geométrica del inverso de la probabilidad que el modelo le dio a la letra correcta. Si un modelo repartiera su probabilidad **uniformemente entre $k$ opciones** y la correcta siempre estuviera entre ellas, cada factor valdría $1/k$ y la perplejidad sería exactamente $k$. Por eso se lee como el **número efectivo de opciones** entre las que el modelo duda en cada paso: un modelo que no sabe nada (uniforme sobre los 92 símbolos) tiene perplejidad 92; uno perfecto, 1. A veces se reporta $H/\ln 2$, los **bits por carácter**: el tamaño, en bits, al que un compresor basado en el modelo reduciría cada letra.
#
# El modelo más simple que aprende algo ignora el contexto y usa la frecuencia de cada carácter (un **unigrama**):

# %%
def perplejidad(log_probs):
    """exp de la entropía cruzada promedio (log_probs: logaritmos naturales de P(letra correcta))."""
    return float(np.exp(-np.mean(log_probs)))


p_uni = (np.bincount(ids_tr, minlength=V) + 1) / (len(ids_tr) + V)   # con +1: hay símbolos que no aparecen en entrenamiento
print(f"uniforme sobre {V} símbolos: perplejidad {V}")
print(f"unigrama (frecuencias):     perplejidad {perplejidad(np.log(p_uni[ids_va[objetivos_va]])):.1f}")

# %% [markdown]
# Conocer solo las frecuencias reduce las «opciones efectivas» de 92 a 21. Para bajar más hay que mirar el contexto.
#
# ## 3. La línea base: $n$-gramas
#
# Un modelo de **$n$-gramas** supone que solo importan los últimos $n-1$ caracteres: $P(x_t \mid x_{<t}) \approx P(x_t \mid x_{t-n+1}, \dots, x_{t-1})$, una cadena de Markov de orden $n-1$. El estimador de máxima verosimilitud es una proporción de conteos, $N(\text{contexto}, x)/N(\text{contexto})$. Y tiene el mismo problema que vimos con Bayes ingenuo en la Unidad 4: un contexto que nunca vimos seguido de cierta letra recibe probabilidad **cero**, y una sola letra con probabilidad cero en validación hace la perplejidad infinita. La solución es la misma: el **suavizado de Laplace** (o aditivo), que suma $\alpha$ a cada conteo:
#
# $$\hat P(x \mid \text{contexto}) = \frac{N(\text{contexto}, x) + \alpha}{N(\text{contexto}) + \alpha V}.$$
#
# Con $\alpha = 1$ es el de Laplace. Para contar rápido, cada $n$-grama se convierte en un entero en base $V$ (con $n \le 7$ cabe en 64 bits).

# %%
def codigos(s, k):
    """Código entero (base V) del k-grama que termina en cada posición i ≥ k-1 de s."""
    c = np.zeros(len(s) - k + 1, dtype=np.int64)
    for j in range(k):
        c = c * V + s[j:len(s) - k + 1 + j]
    return c


class NGramas:
    """Modelo de n-gramas de caracteres con suavizado aditivo."""

    def __init__(self, n, alfa=1.0):
        self.n, self.alfa = n, alfa
        self.gramas, self.n_gramas = np.unique(codigos(ids_tr, n), return_counts=True)
        if n > 1:
            self.contextos, self.n_contextos = np.unique(codigos(ids_tr[:-1], n - 1), return_counts=True)

    @staticmethod
    def _contar(tabla, cuentas, claves):
        i = np.minimum(np.searchsorted(tabla, claves), len(tabla) - 1)
        return np.where(tabla[i] == claves, cuentas[i], 0)

    def log_prob(self, s, posiciones):
        """ln P(s[i] | los n-1 anteriores) para cada i en posiciones (i ≥ n-1)."""
        g = codigos(s, self.n)[posiciones - self.n + 1]
        ng = self._contar(self.gramas, self.n_gramas, g)
        nc = self._contar(self.contextos, self.n_contextos, g // V) if self.n > 1 else len(ids_tr)
        return np.log((ng + self.alfa) / (nc + self.alfa * V))


pos_tr = np.arange(6, len(ids_tr))   # en entrenamiento, todas las posiciones con contexto completo
bigrama = NGramas(2, alfa=1.0)
print(f"bigrama con Laplace: perplejidad de validación {perplejidad(bigrama.log_prob(ids_va, objetivos_va)):.2f}")

# %% [markdown]
# Mirando solo la letra anterior la perplejidad baja a 10. El bigrama sabe, por ejemplo, que después de una «q» viene casi siempre una «u», mientras que después de un espacio puede empezar casi cualquier palabra:

# %%
def siguiente(modelo, letra):
    """Distribución P(· | letra) del bigrama: evalúa los V pares (letra, x)."""
    pares = np.column_stack([np.full(V, a_indice[letra]), np.arange(V)]).ravel()
    return np.exp(modelo.log_prob(pares, np.arange(1, 2 * V, 2)))


fig, ejes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, letra in zip(ejes, ["q", " "]):
    p = siguiente(bigrama, letra)
    orden = np.argsort(p)[::-1][:12]
    ax.bar(range(12), p[orden], color=AZUL)
    ax.set_xticks(range(12), [visible(vocabulario[i]) for i in orden], fontsize=13)
    h = -np.sum(p * np.log(p))
    ax.set_title(f"después de «{visible(letra)}»: perplejidad local e^H = {np.exp(h):.2f}")
    ax.grid(axis="x", visible=False)
for ax in ejes:
    ax.set_ylabel("P(siguiente letra)")
fig.tight_layout()
plt.show()

# %% [markdown]
# La perplejidad «local» $e^{H}$ de cada distribución confirma la lectura de «número efectivo de opciones»: después de «q» hay casi una sola opción (1.04: la «u» se lleva el 99.7 % de la probabilidad); después de un espacio, una veintena (21.8).
#
# ### Contextos más largos: más memoria, peor generalización
#
# Si una letra de contexto ayuda tanto, ¿por qué no usar seis? Probemos $n = 1, \dots, 7$ con dos valores de $\alpha$: el de Laplace y uno mucho menor.

# %%
filas = []
for n in range(1, 8):
    for alfa in [1.0, 0.01]:
        m = NGramas(n, alfa)
        lp_va = m.log_prob(ids_va, objetivos_va)
        filas.append({"n": n, "alfa": alfa,
                      "perplejidad entrenamiento": perplejidad(m.log_prob(ids_tr, pos_tr)),
                      "perplejidad validación": perplejidad(lp_va),
                      "n-gramas distintos": len(m.gramas),
                      "% de n-gramas de validación nunca vistos": 100 * np.mean(
                          NGramas._contar(m.gramas, m.n_gramas, codigos(ids_va, n)[objetivos_va - n + 1]) == 0)})
ngramas = pd.DataFrame(filas)
ngramas.pivot(index="n", columns="alfa", values=["perplejidad entrenamiento", "perplejidad validación"]).round(2)

# %%
fig, ax = plt.subplots(figsize=(10, 4.8))
for alfa, color in [(1.0, ROJO), (0.01, AZUL)]:
    d = ngramas[ngramas.alfa == alfa]
    ax.plot(d.n, d["perplejidad validación"], marker="o", color=color)
    ax.plot(d.n, d["perplejidad entrenamiento"], marker="o", color=color, ls="--", alpha=0.6)
rdp.estilo.etiquetar(ax, 7, {"validación, α = 1": ngramas.query("n == 7 and alfa == 1")["perplejidad validación"].item(),
                             "entrenamiento, α = 1": ngramas.query("n == 7 and alfa == 1")["perplejidad entrenamiento"].item(),
                             "validación, α = 0.01": ngramas.query("n == 7 and alfa == 0.01")["perplejidad validación"].item(),
                             "entrenamiento, α = 0.01": ngramas.query("n == 7 and alfa == 0.01")["perplejidad entrenamiento"].item()})
ax.set_yscale("log")
ax.set_yticks([2, 3, 4, 5, 7, 10, 15, 20], ["2", "3", "4", "5", "7", "10", "15", "20"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("n (el contexto son los n − 1 caracteres anteriores)")
ax.set_ylabel("perplejidad")
ax.set_title("n-gramas de caracteres: el contexto largo se memoriza, no se generaliza")
ax.set_xlim(0.7, 8.8)
plt.show()
ngramas.query("alfa == 0.01")[["n", "n-gramas distintos", "% de n-gramas de validación nunca vistos"]].set_index("n").round(1)

# %% [markdown]
# Es la curva de sobreajuste de la Unidad 3, con $n$ en el papel de la complejidad. En entrenamiento la perplejidad baja siempre al alargar el contexto. En validación baja hasta $n = 5$ (4.40, con $\alpha = 0.01$) y luego sube: el 9.1 % de los 6-gramas de validación y el 18.1 % de los 7-gramas **nunca aparecieron** en 1.9 millones de letras de entrenamiento, y para ellos el modelo no tiene nada que decir más que el suavizado uniforme. El número de contextos posibles crece como $V^{n-1}$; los datos, no. Con $\alpha = 1$ el problema es peor (el mejor es $n = 4$, con 5.74): con contextos largos casi todos los conteos son pequeños y sumarle 1 a cada una de las 92 letras posibles le roba demasiada probabilidad a lo observado; incluso la perplejidad de entrenamiento sube.
#
# Nótese que $\alpha$ es un hiperparámetro y aquí lo elegimos mirando la validación; con rigor habría que reservar un tercer conjunto (Unidad 3). Usaremos dos líneas base: el **bigrama de Laplace** y el **mejor $n$-grama** ($n = 5$, $\alpha = 0.01$), que es un rival serio.
#
# El defecto de fondo es que un $n$-grama trata «dijo don Quijot» y «dijo don Quijo» como contextos **sin relación**: cada uno tiene su propia fila de conteos. Lo que queremos es un modelo que **comparta información** entre contextos parecidos y que pueda usar un contexto largo sin necesitar haberlo visto completo. Eso es lo que hacen las redes neuronales: representan el contexto con un vector, y contextos parecidos dan vectores parecidos.

# %%
mejor_ngrama = NGramas(5, alfa=0.01)
pp_bigrama = perplejidad(bigrama.log_prob(ids_va, objetivos_va))
pp_ngrama = perplejidad(mejor_ngrama.log_prob(ids_va, objetivos_va))
print(f"bigrama (α = 1): {pp_bigrama:.2f}   5-grama (α = 0.01): {pp_ngrama:.2f}")

# %% [markdown]
# ## 4. Redes recurrentes
#
# Una **red recurrente** (RNN) lee el texto letra por letra y mantiene un **estado oculto** $\mathbf{h}_t \in \mathbb{R}^m$ que resume lo leído:
#
# $$\mathbf{h}_t = \tanh\big(W\mathbf{h}_{t-1} + U\mathbf{e}_{x_t} + \mathbf{b}\big), \qquad P(x_{t+1} \mid x_{\le t}) = \operatorname{softmax}(A\mathbf{h}_t + \mathbf{c}),$$
#
# donde $\mathbf{e}_{x_t}$ es el **encaje** (*embedding*) aprendido de la letra $x_t$: una fila de una matriz de $V \times d$, que es lo mismo que multiplicar el vector *one-hot* de la letra por esa matriz. Las mismas matrices $W, U, A$ se usan en **todas** las posiciones: el número de parámetros no depende de la longitud del texto, igual que una convolución comparte su filtro en todas las posiciones de una imagen.
#
# ### Retropropagación en el tiempo y el producto de jacobianas
#
# Para entrenar se «desenrolla» la red sobre una ventana de $T$ letras y se aplica la retropropagación al grafo resultante, que es una red profunda de $T$ capas con pesos compartidos (*backpropagation through time*). El gradiente de la pérdida en la posición $T$ respecto del estado en una posición anterior $t$ es, por la regla de la cadena,
#
# $$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{s=t+1}^{T} \frac{\partial \mathbf{h}_s}{\partial \mathbf{h}_{s-1}} = \prod_{s=t+1}^{T} D_s W, \qquad D_s = \operatorname{diag}\big(1 - \mathbf{h}_s^2\big).$$
#
# Como $0 < 1 - h^2 \le 1$, cada factor tiene norma a lo más $\|W\|_2 = \sigma_{\max}(W)$, y entonces
#
# $$\Big\|\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t}\Big\|_2 \le \sigma_{\max}(W)^{\,T-t}.$$
#
# Si $\sigma_{\max}(W) < 1$, la influencia de una letra sobre el gradiente **decae exponencialmente** con la distancia: el gradiente se **desvanece** y la red no aprende dependencias largas. Si los valores singulares de $W$ son mayores que 1 (y las $\tanh$ no están saturadas), el producto puede **explotar**. Es el mismo fenómeno que con las redes muy profundas de la libreta 1, agravado porque aquí se multiplica **la misma** matriz una y otra vez.
#
# Veámoslo en una RNN de 256 unidades con $W = \rho\, Q$, donde $Q$ es ortogonal (todos sus valores singulares son 1, así que todos los de $W$ valen $\rho$), con entradas al azar:

# %%
def norma_jacobiana_rnn(rho, m=256, pasos=100, semilla=SEMILLA):
    """‖∂h_T/∂h_t‖ (norma de Frobenius / √m) para t = T, T-1, …, 0 en una RNN tanh con W = ρQ."""
    g = torch.Generator().manual_seed(semilla)
    Q, _ = torch.linalg.qr(torch.randn(m, m, generator=g))
    W = rho * Q
    entradas = 0.5 * torch.randn(pasos, m, generator=g)
    h, D = torch.zeros(m), []
    for t in range(pasos):
        h = torch.tanh(W @ h + entradas[t])
        D.append(1 - h ** 2)
    J, normas = torch.eye(m), [1.0]
    for s in range(pasos - 1, 0, -1):          # J = ∂h_T/∂h_s, acumulado hacia atrás
        J = J @ (D[s][:, None] * W)
        normas.append((J.norm() / math.sqrt(m)).item())
    return np.array(normas)


fig, ax = plt.subplots(figsize=(10, 4.6))
normas_rnn = {}
for rho, color in [(0.8, ROJO), (1.0, GRIS), (1.5, DORADO_OSC), (3.0, AZUL)]:
    n = normas_rnn[rho] = norma_jacobiana_rnn(rho)
    ax.plot(np.arange(len(n)), n, color=color)
    ax.text(len(n) + 1, n[-1], f"ρ = {rho}", color=TINTA, va="center", fontsize=11)
ax.set_yscale("log")
ax.set_xlim(0, 112)
ax.set_xlabel("distancia T − t (letras hacia atrás)")
ax.set_ylabel("‖∂h_T / ∂h_t‖ (escala log)")
ax.set_title("RNN con W = ρQ: el gradiente se desvanece o crece exponencialmente con la distancia")
plt.show()

# %% [markdown]
# Con $\rho = 0.8$ la norma cae a $10^{-10}$ a 50 letras de distancia y a $10^{-20}$ a 100. Con $\rho = 1$, aunque $W$ no encoge ninguna dirección, las derivadas de $\tanh$ (todas menores que 1) bastan para que caiga a $10^{-6}$ a 50 letras. Con $\rho = 1.5$ se mantiene del orden de 1: el crecimiento por $W$ y la atenuación por las $\tanh$ casi se compensan. Con $\rho = 3$ explota: $10^{6}$ a 50 letras y $10^{13}$ a 100. La franja de valores en la que el gradiente ni muere ni explota es angosta, y además depende de cuánto se saturan las $\tanh$, es decir, de los datos. La RNN simple vive en el filo de la navaja, y en la práctica casi siempre cae del lado del desvanecimiento: aprende bien dependencias de unas cuantas letras y casi nada más lejos.
#
# **Recorte de gradiente.** Contra la explosión hay un remedio simple y muy usado: si la norma del gradiente completo $\mathbf{g}$ pasa de un umbral $c$, se reescala, $\mathbf{g} \leftarrow \mathbf{g}\cdot\min(1, c/\|\mathbf{g}\|)$. La dirección se conserva y el paso no se dispara. Contra el desvanecimiento no basta con reescalar: hay que cambiar la arquitectura.
#
# ## 5. LSTM: una autopista para el gradiente
#
# La **LSTM** (*long short-term memory*; Hochreiter y Schmidhuber, 1997) agrega una **celda** de memoria $\mathbf{c}_t$ que se actualiza de forma **aditiva**, controlada por tres **compuertas** con valores en $(0, 1)$:
#
# $$
# \begin{aligned}
# \mathbf{f}_t &= \sigma(W_f\mathbf{h}_{t-1} + U_f\mathbf{e}_{x_t} + \mathbf{b}_f) && \text{(olvido)}\\
# \mathbf{i}_t &= \sigma(W_i\mathbf{h}_{t-1} + U_i\mathbf{e}_{x_t} + \mathbf{b}_i) && \text{(entrada)}\\
# \mathbf{o}_t &= \sigma(W_o\mathbf{h}_{t-1} + U_o\mathbf{e}_{x_t} + \mathbf{b}_o) && \text{(salida)}\\
# \tilde{\mathbf{c}}_t &= \tanh(W_c\mathbf{h}_{t-1} + U_c\mathbf{e}_{x_t} + \mathbf{b}_c) && \text{(candidato)}\\
# \mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t, \qquad \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t).
# \end{aligned}
# $$
#
# La clave está en la penúltima línea. Si ignoramos la dependencia indirecta a través de $\mathbf{h}_{t-1}$, $\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1} = \operatorname{diag}(\mathbf{f}_t)$: **no hay matriz de pesos ni $\tanh$** en ese camino. Si la red aprende a dejar la compuerta de olvido cerca de 1 en alguna unidad, el gradiente viaja hacia atrás por esa unidad casi sin atenuarse: una «autopista» (la misma idea que las conexiones residuales de la libreta 3: $\partial(\mathbf{x} + F(\mathbf{x}))/\partial\mathbf{x} = I + \cdots$). La red **decide** qué recordar y qué olvidar, y lo aprende de los datos.
#
# ### El modelo y el entrenamiento
#
# Usamos el `nn.LSTM` de PyTorch con encajes de dimensión 64 y estado de 256. El entrenamiento toma ventanas de $T = 128$ letras en posiciones al azar del texto de entrenamiento; en cada ventana la red predice las 128 letras siguientes a la vez (la entrada desplazada una posición). La pérdida es la entropía cruzada promedio. (Aquí se ve la razón de calificar solo letras con al menos 64 de contexto: la red solo ve lo que hay dentro de su ventana, y una letra del principio de la ventana tendría casi nada de contexto; mediríamos el corte de la ventana, no el modelo.) Usamos AdamW con un calendario de calentamiento corto y coseno (libreta 2), y recorte de gradiente con $c = 1$.

# %%
datos_tr = torch.from_numpy(ids_tr).to(dispositivo)
datos_va = torch.from_numpy(ids_va).to(dispositivo)
B = 32                       # ventanas por minilote
CALENTAMIENTO = 50           # pasos de calentamiento de la tasa de aprendizaje
RECORTE = 1.0                # umbral del recorte de gradiente


def lote(generador):
    """B ventanas al azar del texto de entrenamiento: entrada x y objetivo y (x desplazado una letra)."""
    i = torch.randint(0, len(ids_tr) - T - 1, (B,), generator=generador)
    v = torch.stack([torch.from_numpy(ids_tr[j:j + T + 1]) for j in i.tolist()]).to(dispositivo)
    return v[:, :-1], v[:, 1:]


@torch.no_grad()
def evaluar(modelo, ventanas=None):
    """Perplejidad en validación sobre las letras calificadas (las últimas 64 de cada ventana)."""
    modelo.eval()
    inicios = inicios_va if ventanas is None else inicios_va[ventanas]
    total = 0.0
    for k in range(0, len(inicios), 128):
        v = torch.stack([datos_va[j:j + T + 1] for j in inicios[k:k + 128]])
        logits = modelo(v[:, :-1])[:, -CONTEXTO_MIN:]
        total += F.cross_entropy(logits.reshape(-1, V), v[:, -CONTEXTO_MIN:].reshape(-1), reduction="sum").item()
    modelo.train()
    return math.exp(total / (len(inicios) * CONTEXTO_MIN))


def entrenar(modelo, pasos, tasa, cada=100):
    """Entrena con AdamW, calentamiento + coseno y recorte de gradiente; devuelve la historia."""
    opt = torch.optim.AdamW(modelo.parameters(), lr=tasa, weight_decay=0.01)
    calendario = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / CALENTAMIENTO) * 0.5 * (1 + math.cos(math.pi * t / pasos)))
    generador = torch.Generator().manual_seed(SEMILLA)
    muestra_va = np.arange(0, len(inicios_va), 8)          # 1 de cada 8 ventanas, para la curva
    historia, normas, inicio = [], [], time.time()
    for paso in range(1, pasos + 1):
        x, y = lote(generador)
        perdida = F.cross_entropy(modelo(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        perdida.backward()
        normas.append(nn.utils.clip_grad_norm_(modelo.parameters(), RECORTE).item())  # norma antes del recorte
        opt.step()
        calendario.step()
        if paso % cada == 0 or paso == pasos:
            historia.append({"paso": paso, "entrenamiento": math.exp(perdida.item()),
                             "validación": evaluar(modelo, muestra_va)})
    print(f"{pasos} pasos en {time.time() - inicio:.0f} s")
    return pd.DataFrame(historia).set_index("paso"), np.array(normas)


class LSTMLenguaje(nn.Module):
    def __init__(self, V, d=64, m=256):
        super().__init__()
        self.encaje = nn.Embedding(V, d)
        self.lstm = nn.LSTM(d, m, batch_first=True)
        self.salida = nn.Linear(m, V)

    def forward(self, x):                       # x: (B, T) índices → (B, T, V) logits
        h, _ = self.lstm(self.encaje(x))
        return self.salida(h)


PASOS_LSTM = 120 if RAPIDO else 1200
torch.manual_seed(SEMILLA)
lstm = LSTMLenguaje(V).to(dispositivo)
lstm_inicial = LSTMLenguaje(V).to(dispositivo)
lstm_inicial.load_state_dict(lstm.state_dict())          # copia sin entrenar, para comparar después
print(f"parámetros de la LSTM: {sum(p.numel() for p in lstm.parameters()):,}")
historia_lstm, normas_lstm = entrenar(lstm, PASOS_LSTM, tasa=0.02)
pp_lstm = evaluar(lstm)
print(f"LSTM: perplejidad de validación {pp_lstm:.2f}")

# %% [markdown]
# ¿Qué tanto trabajó el recorte? Veamos la norma del gradiente completo antes de recortar, paso por paso:

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.arange(1, len(normas_lstm) + 1), normas_lstm, color=AZUL, lw=0.9)
ax.axhline(RECORTE, color=ROJO, ls="--", lw=1.4)
ax.text(len(normas_lstm) * 0.99, RECORTE * 1.08, "umbral de recorte c = 1", color=ROJO, ha="right", va="bottom")
ax.set_yscale("log")
ax.set_xlabel("paso de entrenamiento")
ax.set_ylabel("‖g‖ antes del recorte")
ax.set_title("Norma del gradiente durante el entrenamiento de la LSTM")
plt.show()
print(f"pasos recortados: {np.sum(normas_lstm > RECORTE)} de {len(normas_lstm)}; "
      f"norma máxima {normas_lstm.max():.2f} (paso {normas_lstm.argmax() + 1}); mediana {np.median(normas_lstm):.2f}")

# %% [markdown]
# La norma típica es pequeña (mediana 0.13) y el recorte solo actuó en 2 de los 1,200 pasos, ambos al principio (el máximo, 1.29, en el paso 7), cuando los pesos recién inicializados producen gradientes grandes. En esta LSTM el recorte es un seguro barato más que una necesidad: la celda aditiva (que veremos enseguida en acción) la hace mucho menos propensa a explotar que una RNN simple. Pero un solo paso descontrolado puede arruinar un entrenamiento de horas, y protegerse cuesta una línea de código.

# %% [markdown]
# ### ¿Llega el gradiente lejos?
#
# Midamos directamente lo que la teoría predice: tomemos ventanas de validación, la pérdida **de la última letra** y la norma de su gradiente respecto del encaje de cada letra anterior, $\|\partial\ell_T/\partial\mathbf{e}_{x_t}\|$, como función de la distancia $T - t$. Lo comparamos en la LSTM recién inicializada y en la entrenada.

# %%
def gradiente_por_distancia(modelo, n_ventanas=64):
    """Promedio de ‖∂ℓ_T/∂e_t‖ para t = T, T-1, …, 1 (ℓ_T: pérdida de la última letra de la ventana)."""
    modelo.eval()
    v = torch.stack([datos_va[j:j + T + 1] for j in inicios_va[:n_ventanas]])
    e = modelo.encaje(v[:, :-1]).detach().requires_grad_(True)
    if isinstance(modelo, LSTMLenguaje):
        logits = modelo.salida(modelo.lstm(e)[0])
    else:
        logits = modelo.desde_encajes(e)
    F.cross_entropy(logits[:, -1], v[:, -1], reduction="sum").backward()
    modelo.train()
    return e.grad.norm(dim=-1).mean(0).flip(0).cpu().numpy() / n_ventanas


def relativa(g):
    """Normaliza por el valor en t = T; los ceros exactos (subdesbordamiento) se omiten en la escala log."""
    r = g / g[0]
    return np.where(r > 0, r, np.nan)


g_ini, g_ent = gradiente_por_distancia(lstm_inicial), gradiente_por_distancia(lstm)
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(relativa(g_ini), color=GRIS)
ax.plot(relativa(g_ent), color=AZUL)
ax.set_yscale("log")
ax.text(40, relativa(g_ini)[40] * 30, "LSTM sin entrenar", color=GRIS, fontsize=12)
ax.text(70, relativa(g_ent)[70] * 8, "LSTM entrenada", color=AZUL, fontsize=12)
ax.set_xlim(0, T)
ax.set_xlabel("distancia T − t (letras hacia atrás)")
ax.set_ylabel("‖∂ℓ_T / ∂e_t‖, relativa a t = T")
ax.set_title("Cuánto influye cada letra anterior en el gradiente de la última predicción")
plt.show()
for d in [10, 30, 60, 120]:
    print(f"distancia {d:3d}: sin entrenar {g_ini[d] / g_ini[0]:.1e}   entrenada {g_ent[d] / g_ent[0]:.1e}")

# %% [markdown]
# Sin entrenar, la compuerta de olvido vale alrededor de $\sigma(0) = 0.5$ y la LSTM se comporta como una RNN con $\rho < 1$: la influencia relativa cae a $10^{-6}$ a 30 letras y a $10^{-12}$ a 60 (a 120 el número es tan pequeño que en punto flotante se vuelve cero). Entrenada, la caída es muy distinta: a 30 letras es $4.5\times10^{-4}$ y luego casi se estabiliza, $4.1\times10^{-4}$ a 60 y $1.5\times10^{-5}$ a 120. La red aprendió a abrir la autopista en algunas unidades. Aun así, cada letra lejana influye miles de veces menos que la última, y su información tiene que llegar «pasando» por todo el estado intermedio. Ésa es la limitación que ataca la atención.

# %% [markdown]
# ## 6. Atención
#
# La LSTM tiene que comprimir todo el pasado en un vector de 256 números y llevarlo paso a paso. La **atención** (Bahdanau, Cho y Bengio, 2015, para traducción automática) propone otra cosa: cuando hace falta un dato del pasado, **ir a buscarlo directamente**.
#
# ### Una búsqueda suave en un diccionario
#
# Un diccionario de Python guarda pares (clave, valor); una búsqueda compara la **consulta** con cada clave y devuelve el valor de la que coincide. La atención hace lo mismo, pero en versión suave y diferenciable: cada posición $s$ del pasado ofrece una clave $\mathbf{k}_s$ y un valor $\mathbf{v}_s$; la posición actual hace una consulta $\mathbf{q}_t$, mide la similitud con cada clave por producto punto y devuelve un **promedio ponderado** de los valores:
#
# $$\alpha_{ts} = \frac{\exp(\mathbf{q}_t^\top\mathbf{k}_s/\sqrt{d_k})}{\sum_{s'} \exp(\mathbf{q}_t^\top\mathbf{k}_{s'}/\sqrt{d_k})}, \qquad \text{salida}_t = \sum_s \alpha_{ts}\,\mathbf{v}_s .$$
#
# En la **autoatención**, consultas, claves y valores salen de la misma secuencia, por tres proyecciones lineales aprendidas: $Q = XW_Q$, $K = XW_K$, $V = XW_V$, con $X \in \mathbb{R}^{T\times d}$ las representaciones de las $T$ posiciones. En forma matricial, para todas las posiciones a la vez,
#
# $$\operatorname{Atención}(Q, K, V) = \operatorname{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}} + M\Big)V,$$
#
# donde $M$ es la **máscara causal**: $M_{ts} = -\infty$ si $s > t$ y 0 si no. Con ella, la posición $t$ no puede «ver el futuro» (que es justo lo que tiene que predecir), y una sola pasada calcula las $T$ predicciones de la ventana en paralelo. El camino entre dos posiciones cualesquiera tiene longitud 1: el gradiente no tiene que atravesar $T - t$ jacobianas.
#
# ### ¿Por qué dividir entre $\sqrt{d_k}$?
#
# Supongamos que, al inicio del entrenamiento, las componentes de $\mathbf{q}$ y $\mathbf{k}$ son independientes, con media 0 y varianza 1. Entonces cada término $q_i k_i$ tiene media 0 y varianza $\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1$, y como los $d_k$ términos son independientes,
#
# $$\mathbb{E}[\mathbf{q}^\top\mathbf{k}] = 0, \qquad \operatorname{Var}(\mathbf{q}^\top\mathbf{k}) = \sum_{i=1}^{d_k}\operatorname{Var}(q_ik_i) = d_k .$$
#
# Los puntajes tienen desviación estándar $\sqrt{d_k}$: con $d_k = 64$, típicamente $\pm 8$, y la diferencia entre el mayor y los demás es tan grande que el softmax se vuelve casi un *one-hot*. Eso es malo para aprender: la jacobiana del softmax es $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$, que tiende a la matriz cero cuando $\mathbf{p}$ se acerca a un vector canónico. Dividir entre $\sqrt{d_k}$ devuelve los puntajes a varianza 1, sin importar la dimensión. Verifiquémoslo:

# %%
gen = torch.Generator().manual_seed(SEMILLA)
filas = []
for dk in [4, 16, 64, 256, 1024]:
    q = torch.randn(2000, 1, dk, generator=gen)
    k = torch.randn(2000, 64, dk, generator=gen)              # 64 claves por consulta
    puntajes = (q * k).sum(-1)
    for escala, nombre in [(1.0, "sin escalar"), (math.sqrt(dk), "entre √d_k")]:
        p = torch.softmax(puntajes / escala, -1)
        jac = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
        filas.append({"d_k": dk, "escala": nombre, "desv. est. del puntaje": (puntajes / escala).std().item(),
                      "peso máximo promedio": p.max(-1).values.mean().item(),
                      "‖jacobiana del softmax‖": jac.norm(dim=(1, 2)).mean().item()})
escala_raiz = pd.DataFrame(filas)
escala_raiz.pivot(index="d_k", columns="escala").round(3)

# %%
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
for nombre, color in [("sin escalar", ROJO), ("entre √d_k", AZUL)]:
    d = escala_raiz[escala_raiz.escala == nombre]
    ejes[0].plot(d.d_k, d["peso máximo promedio"], marker="o", color=color)
    ejes[1].plot(d.d_k, d["‖jacobiana del softmax‖"], marker="o", color=color)
for ax, col in zip(ejes, ["peso máximo promedio", "‖jacobiana del softmax‖"]):
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d_k")
    rdp.estilo.etiquetar(ax, 1024, {n: escala_raiz.query("d_k == 1024 and escala == @n")[col].item() for n in ["sin escalar", "entre √d_k"]})
    ax.set_xlim(3, 4000)
ejes[0].set_title("Peso de la clave más parecida (64 claves)")
ejes[1].set_title("Tamaño de la jacobiana del softmax")
ejes[0].set_ylim(0, 1.05)
fig.tight_layout()
plt.show()

# %% [markdown]
# Sin escalar, la desviación estándar de los puntajes es $\sqrt{d_k}$ (2, 4, 8, 16 y 32), como predice la derivación. Con $d_k = 1024$ el softmax pone en promedio el 95 % del peso en una sola clave y su jacobiana es casi tres veces más chica que con la escala (0.065 contra 0.18): la atención se vuelve una búsqueda «dura» desde el primer paso y el gradiente casi no fluye por ella. Escalando, el comportamiento es el mismo para cualquier $d_k$. (Es el mismo razonamiento de varianzas que da la inicialización de Xavier y He en la libreta 1.)
#
# ### Varias cabezas, posición y el bloque *transformer*
#
# - **Varias cabezas.** En vez de una atención con $d_k = d$, se hacen $H$ atenciones en paralelo con $d_k = d/H$, cada una con sus propias proyecciones, y se concatenan sus salidas. Cada cabeza puede especializarse en una relación distinta (la letra anterior, el inicio de la palabra, la última raya de diálogo…) con el mismo costo total.
# - **Posición.** La atención por sí sola no sabe en qué orden están las letras: si se permutan las posiciones de $X$, las salidas se permutan igual (es *equivariante* a permutaciones; ejercicio 4). Hay que inyectar la posición. Vaswani et al. (2017) suman a cada encaje un vector de senos y cosenos de distintas frecuencias, $\mathrm{PE}_{(p, 2i)} = \sin(p/10000^{2i/d})$, $\mathrm{PE}_{(p, 2i+1)} = \cos(p/10000^{2i/d})$; aquí, como en GPT, usamos un **encaje de posición aprendido**: una tabla de $T \times d$ que se suma al encaje de la letra.
# - **El bloque.** Un bloque *transformer* (Vaswani et al., 2017) alterna autoatención y un MLP aplicado a cada posición por separado, cada uno con una **conexión residual** y una **normalización de capa** (que estandariza cada vector de $d$ componentes):
#
# $$\mathbf{x} \leftarrow \mathbf{x} + \operatorname{AtenciónMulticabeza}\big(\operatorname{NormCapa}(\mathbf{x})\big), \qquad \mathbf{x} \leftarrow \mathbf{x} + \operatorname{MLP}\big(\operatorname{NormCapa}(\mathbf{x})\big).$$
#
# La atención **mezcla información entre posiciones**; el MLP **procesa cada posición**. Las conexiones residuales son, otra vez, la autopista del gradiente.
#
# **Costo.** La matriz $QK^\top$ es de $T\times T$: la atención cuesta $O(T^2 d)$ operaciones y $O(T^2)$ memoria por capa, contra $O(T d^2)$ de las proyecciones y el MLP. Con contextos cortos domina el MLP; con contextos de decenas de miles de letras, la atención. A cambio, a diferencia de la RNN, todas las posiciones se calculan **en paralelo**, que es lo que aprovechan las GPU.
#
# ## 7. Un *transformer* mínimo
#
# Escribimos la autoatención explícitamente, sin usar los módulos de PyTorch que la traen hecha, para que se vea la fórmula: $d = 128$, 4 cabezas ($d_k = 32$), 2 bloques, contexto de 128 letras.

# %%
class AutoatencionCausal(nn.Module):
    def __init__(self, d, cabezas):
        super().__init__()
        self.cabezas, self.dk = cabezas, d // cabezas
        self.qkv = nn.Linear(d, 3 * d)           # W_Q, W_K, W_V de todas las cabezas juntas
        self.proyeccion = nn.Linear(d, d)
        self.guardar_pesos, self.pesos = False, None

    def forward(self, x):                       # x: (B, T, d)
        B_, T_, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (z.view(B_, T_, self.cabezas, self.dk).transpose(1, 2) for z in (q, k, v))  # (B, H, T, d_k)
        puntajes = q @ k.transpose(-2, -1) / math.sqrt(self.dk)                              # (B, H, T, T)
        futuro = torch.triu(torch.ones(T_, T_, dtype=torch.bool, device=x.device), diagonal=1)
        alfa = puntajes.masked_fill(futuro, float("-inf")).softmax(dim=-1)                   # máscara causal
        if self.guardar_pesos:
            self.pesos = alfa.detach()
        y = (alfa @ v).transpose(1, 2).reshape(B_, T_, d)                                    # concatena las cabezas
        return self.proyeccion(y)


class BloqueTransformer(nn.Module):
    def __init__(self, d, cabezas):
        super().__init__()
        self.norma1, self.norma2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.atencion = AutoatencionCausal(d, cabezas)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.atencion(self.norma1(x))      # mezcla entre posiciones
        return x + self.mlp(self.norma2(x))        # procesa cada posición


class TransformerLenguaje(nn.Module):
    def __init__(self, V, d=128, cabezas=4, bloques=2, contexto=T):
        super().__init__()
        self.encaje = nn.Embedding(V, d)
        self.posicion = nn.Embedding(contexto, d)
        self.bloques = nn.ModuleList([BloqueTransformer(d, cabezas) for _ in range(bloques)])
        self.norma = nn.LayerNorm(d)
        self.salida = nn.Linear(d, V)

    def desde_encajes(self, e):                 # e: (B, T, d) encajes de las letras
        x = e + self.posicion(torch.arange(e.shape[1], device=e.device))
        for bloque in self.bloques:
            x = bloque(x)
        return self.salida(self.norma(x))

    def forward(self, x):
        return self.desde_encajes(self.encaje(x))


PASOS_TF = 150 if RAPIDO else 1500
torch.manual_seed(SEMILLA)
transformer = TransformerLenguaje(V).to(dispositivo)
print(f"parámetros del transformer: {sum(p.numel() for p in transformer.parameters()):,}")
historia_tf, normas_tf = entrenar(transformer, PASOS_TF, tasa=0.008)
pp_tf = evaluar(transformer)
print(f"transformer: perplejidad de validación {pp_tf:.2f}")

# %% [markdown]
# ## 8. Comparación
#
# Todos los modelos se califican en las mismas letras de validación.

# %%
n_param = lambda m: sum(p.numel() for p in m.parameters())
comparacion = pd.DataFrame([
    {"modelo": "uniforme", "parámetros": 0, "perplejidad": float(V)},
    {"modelo": "unigrama", "parámetros": V, "perplejidad": perplejidad(np.log(p_uni[ids_va[objetivos_va]]))},
    {"modelo": "bigrama (Laplace)", "parámetros": V * V, "perplejidad": pp_bigrama},
    {"modelo": "5-grama (α = 0.01)", "parámetros": len(mejor_ngrama.gramas), "perplejidad": pp_ngrama},
    {"modelo": "LSTM", "parámetros": n_param(lstm), "perplejidad": pp_lstm},
    {"modelo": "transformer", "parámetros": n_param(transformer), "perplejidad": pp_tf},
]).set_index("modelo")
comparacion["bits por carácter"] = np.log2(comparacion["perplejidad"])
comparacion.style.format({"parámetros": "{:,}", "perplejidad": "{:.2f}", "bits por carácter": "{:.2f}"})

# %% [markdown]
# (En los $n$-gramas, «parámetros» es el número de conteos distintos de cero que hay que guardar.)

# %%
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(historia_lstm.index, historia_lstm["validación"], marker="o", ms=4, color=AZUL)
ax.plot(historia_tf.index, historia_tf["validación"], marker="o", ms=4, color=ROJO)
ax.axhline(pp_bigrama, color=GRIS, ls=":", lw=1.4)
ax.axhline(pp_ngrama, color=DORADO_OSC, ls="--", lw=1.4)
ax.text(PASOS_TF, pp_bigrama * 1.03, "bigrama", color=GRIS, ha="right", va="bottom")
ax.text(PASOS_TF, pp_ngrama * 1.03, "5-grama", color=DORADO_OSC, ha="right", va="bottom")
ax.text(PASOS_LSTM, historia_lstm["validación"].iloc[-1] * 0.93, "LSTM", color=AZUL, ha="center", va="top")
ax.text(PASOS_TF + 25, historia_tf["validación"].iloc[-1], "transformer", color=ROJO, va="center")
ax.set_yscale("log")
ax.set_yticks([3, 4, 5, 7, 10, 15, 20], ["3", "4", "5", "7", "10", "15", "20"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlim(0, PASOS_TF * 1.18)
ax.set_ylim(3.3, 12)
ax.set_xlabel("paso de entrenamiento (32 ventanas de 128 letras)")
ax.set_ylabel("perplejidad de validación")
ax.set_title("Curvas de validación de las dos redes")
plt.show()

# %% [markdown]
# Las dos redes dejan muy atrás al bigrama (10.02) y también superan al mejor 5-grama (4.40): la LSTM llega a 3.96 y el *transformer* a 3.90, con un número de parámetros (359,260 y 436,828) del mismo orden que los 137,848 conteos que guarda el 5-grama. La diferencia entre las dos redes es pequeña y depende de los hiperparámetros elegidos: con unos minutos de CPU no se puede concluir que una arquitectura sea mejor que la otra en este problema.
#
# Las curvas sí muestran una diferencia de comportamiento: la LSTM aprende mucho más rápido al principio (perplejidad 6.1 a los 100 pasos, contra 10.3 del *transformer*). La recurrencia trae de fábrica la idea de que lo reciente importa más; el *transformer* tiene que descubrir, a partir de los datos, a qué posiciones mirar. Al final, las dos curvas se aplanan en parte porque el calendario coseno lleva la tasa de aprendizaje a cero; para saber cuánto más pueden bajar hay que entrenar más (ejercicio 5).
#
# En bits: $\log_2 3.90 = 1.96$ bits por carácter para el *transformer*, contra $\log_2 92 = 6.5$ del modelo uniforme y los 8 bits de un archivo de texto con un byte por letra. Un compresor que usara este modelo reduciría el texto de validación a la cuarta parte.
#
# ### El camino corto de la atención
#
# Repitamos la medición de la sección 5, ahora con las dos redes entrenadas: cuánto pesa en el gradiente de la última predicción cada letra anterior.

# %%
g_tf = gradiente_por_distancia(transformer)
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(relativa(g_ent), color=AZUL)
ax.plot(relativa(g_tf), color=ROJO)
ax.set_yscale("log")
ax.text(70, relativa(g_ent)[70] * 0.12, "LSTM entrenada", color=AZUL, fontsize=12)
ax.text(70, relativa(g_tf)[70] * 4, "transformer entrenado", color=ROJO, fontsize=12)
ax.set_xlim(0, T)
ax.set_xlabel("distancia T − t (letras hacia atrás)")
ax.set_ylabel("‖∂ℓ_T / ∂e_t‖, relativa a t = T")
ax.set_title("Influencia de cada letra anterior: recurrencia contra atención")
plt.show()
for d in [10, 30, 60, 120]:
    print(f"distancia {d:3d}: LSTM {g_ent[d] / g_ent[0]:.1e}   transformer {g_tf[d] / g_tf[0]:.1e}")

# %% [markdown]
# En el *transformer* la influencia también decae con la distancia (para predecir la siguiente letra, las cercanas importan más), pero mucho más despacio: a 120 letras es $2.8\times10^{-3}$, casi 200 veces más que en la LSTM. Ninguna letra de la ventana está a más de un paso de atención de la que se predice; si el modelo necesita algo de lejos, el gradiente le dice cómo ir a buscarlo.

# %% [markdown]
# ## 9. Generar texto: la temperatura
#
# Con un modelo de lenguaje se puede **generar** texto: se da un inicio, se muestrea la siguiente letra de $P(x_t \mid x_{<t})$, se agrega al contexto y se repite. La **temperatura** $\tau$ modifica la distribución antes de muestrear dividiendo los *logits* $z_i$:
#
# $$p_i(\tau) = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}} .$$
#
# Con $\tau = 1$ se muestrea del modelo tal cual; con $\tau \to 0$ se elige siempre la letra más probable; con $\tau > 1$ la distribución se aplana hacia la uniforme.

# %%
@torch.no_grad()
def generar(modelo, inicio, n=300, temperatura=1.0, semilla=SEMILLA):
    modelo.eval()
    g = torch.Generator(device=dispositivo).manual_seed(semilla)
    x = torch.tensor([a_indice[c] for c in inicio], device=dispositivo)
    for _ in range(n):
        logits = modelo(x[-T:][None])[0, -1]
        p = torch.softmax(logits / temperatura, dim=-1)
        x = torch.cat([x, torch.multinomial(p, 1, generator=g)])
    modelo.train()
    return "".join(vocabulario[i] for i in x.tolist())


INICIO = "—Sancho amigo —dijo don Quijote—,"
generados = {tau: generar(transformer, INICIO, temperatura=tau) for tau in [0.5, 0.8, 1.2]}
for tau, s in generados.items():
    print(f"── temperatura {tau} " + "─" * 40)
    print(s, "\n")

# %% [markdown]
# Con $\tau = 0.5$ el texto es conservador: palabras frecuentes y frases hechas del libro («de la historia», «don Quijote», «de la Mancha»), casi todas palabras reales, aunque las frases no tengan sentido. Con $\tau = 0.8$ hay más variedad y rasgos del estilo de Cervantes, como «vuesa merced» y un nuevo turno de diálogo que empieza con raya, junto con palabras inventadas pero pronunciables («tantado»). Con $\tau = 1.2$ las palabras se descomponen («ságimiente», «tendrómogándose»). Nótese también que las líneas generadas miden entre 60 y 80 caracteres, como en el archivo de Gutenberg (donde la mitad de las líneas mide entre 68 y 74): el modelo aprendió cuándo toca un salto de línea, y en la sección 10 veremos qué cabeza de atención se encarga de eso.
#
# En ningún caso hay **sentido**: el modelo aprendió la forma del texto (ortografía, palabras, algo de sintaxis local, la puntuación del diálogo), no lo que el texto quiere decir. Volveremos a esto al final.
#
# ### Dónde se sorprende el modelo
#
# La perplejidad es un promedio. Mirando la **sorpresa** de cada letra, $-\log_2 P(x_t \mid x_{<t})$ en bits, se ve dónde están las dificultades. Tomemos una frase de validación:

# %%
FRASE = "Hay mucho que decir en eso —dijo Sancho—. Durmamos, por ahora, entrambos"
i0 = texto.index(FRASE, corte)
ventana = torch.from_numpy(ids[i0 + len(FRASE) - T - 1:i0 + len(FRASE)]).to(dispositivo)   # la frase son las últimas letras
with torch.no_grad():
    transformer.eval()
    lp = torch.log_softmax(transformer(ventana[None, :-1])[0], -1)
    transformer.train()
sorpresa = -(lp[torch.arange(T), ventana[1:]] / math.log(2)).cpu().numpy()[-len(FRASE):]

fig, ax = plt.subplots(figsize=(13, 1.3))
norma = plt.Normalize(0, 8)
for j, (c, s) in enumerate(zip(FRASE, sorpresa)):
    ax.add_patch(plt.Rectangle((j, 0), 1, 1, color=rdp.estilo.SECUENCIAL(norma(s))))
    ax.text(j + 0.5, 0.5, visible(c), ha="center", va="center", fontsize=13,
            color="white" if s > 4.5 else TINTA)
ax.set_xlim(0, len(FRASE))
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Sorpresa del transformer en cada letra (bits): más oscuro = más inesperada", loc="left")
plt.show()
print("las 5 letras más sorprendentes:", [(FRASE[j], round(float(sorpresa[j]), 1)) for j in np.argsort(sorpresa)[::-1][:5]])

# %% [markdown]
# En esta frase el promedio es de 1.72 bits por letra. Las letras caras son, sobre todo, **las primeras de cada palabra** (la «q» de «que» cuesta 6.2 bits; la «m» de «mucho», 5.1; la «p» de «por», 4.4) y las de las **palabras poco frecuentes**: «Durmamos» cuesta en total unos 29 bits, y la «m» de «entrambos», 5.6. En cambio, una vez que empieza una palabra común, el resto es casi gratis (la «u» y la «e» de «que» cuestan 0.0 y 0.07 bits), y los espacios son baratos: saber que una palabra terminó es fácil. «—dijo Sancho» sale casi regalado: tras la raya, la «d» cuesta 0.9 bits y el resto de «dijo» casi nada; la única duda real es la «S» (2.1 bits), entre Sancho y don Quijote.
#
# ## 10. ¿Qué miran las cabezas de atención?
#
# Guardemos los pesos $\alpha_{ts}$ de las 8 cabezas (4 por bloque) al procesar la frase anterior, y dibujemos, para cada cabeza, la matriz de pesos de sus últimas 40 posiciones: la fila $t$ dice cuánto mira la letra $t$ a cada letra anterior $s$.

# %%
@torch.no_grad()
def pesos_atencion(modelo, x):
    """Pesos α de todas las cabezas para la entrada x (B, T): arreglo (B, bloques, cabezas, T, T)."""
    for bloque in modelo.bloques:
        bloque.atencion.guardar_pesos = True
    modelo.eval()
    modelo(x)
    modelo.train()
    for bloque in modelo.bloques:
        bloque.atencion.guardar_pesos = False
    return torch.stack([b.atencion.pesos for b in modelo.bloques], dim=1).cpu().numpy()


pesos = pesos_atencion(transformer, ventana[None, 1:])[0]          # (bloques, cabezas, T, T)

L = 40
ultimas = "".join(vocabulario[i] for i in ventana[1:].tolist()[-L:])   # las 40 letras de entrada más recientes
letras = [visible(c) for c in ultimas]
fig, ejes = plt.subplots(2, 4, figsize=(14, 7.4))
for b in range(2):
    for h in range(4):
        ax = ejes[b, h]
        ax.imshow(pesos[b, h, -L:, -L:], cmap=rdp.estilo.SECUENCIAL, vmin=0, vmax=1)
        ax.set_title(f"bloque {b + 1}, cabeza {h + 1}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
fig.suptitle(f"Pesos de atención sobre «…{ultimas}»", x=0.02, ha="left", color=rdp.estilo.MARINO)
fig.tight_layout()
plt.show()

# %% [markdown]
# Cada cabeza tiene un patrón propio. Para no quedarnos con la impresión de una sola frase, resumamos cada cabeza en 64 ventanas de validación (las últimas 64 filas de cada una): cuánto peso pone, en promedio, en la letra anterior y en las de hace dos y tres posiciones, en la primera letra de la palabra que se está escribiendo, en los saltos de línea, y a qué distancia promedio mira, $\sum_s \alpha_{ts}(t - s)$.

# %%
v = torch.stack([datos_va[j:j + T] for j in inicios_va[:64]])
P = pesos_atencion(transformer, v)[..., -CONTEXTO_MIN:, :]          # (64 ventanas, bloques, cabezas, 64 filas, T)
filas_t = np.arange(T - CONTEXTO_MIN, T)
distancia = np.clip(filas_t[:, None] - np.arange(T)[None, :], 0, None)  # t − s (el futuro tiene peso 0)
x = v.cpu().numpy()
separador = np.isin(x, [a_indice[" "], a_indice["\n"]])
tras_separador = np.zeros_like(separador)
tras_separador[:, 1:] = separador[:, :-1]
inicio_palabra = np.maximum.accumulate(np.where(tras_separador, np.arange(T), 0), axis=1)[:, filas_t]
es_salto = (x == a_indice["\n"])[:, None, :]
perfil = []
for b in range(2):
    for h in range(4):
        A = P[:, b, h]
        atras = lambda k: A[:, np.arange(CONTEXTO_MIN), filas_t - k].mean()
        perfil.append({"cabeza": f"bloque {b + 1}, cabeza {h + 1}",
                       "t − 1": atras(1), "t − 2": atras(2), "t − 3": atras(3),
                       "inicio de la palabra": np.take_along_axis(A, inicio_palabra[..., None], -1).mean(),
                       "saltos de línea": (A * es_salto).sum(-1).mean(),
                       "distancia promedio": (A * distancia).sum(-1).mean()})
perfil = pd.DataFrame(perfil).set_index("cabeza")
print(f"fracción de saltos de línea en el texto: {np.mean(ids_va == a_indice[chr(10)]):.3f}")
perfil.round(2)

# %% [markdown]
# Varias cabezas tienen un papel muy claro. En el primer bloque, la 1.2 (bloque 1, cabeza 2) pone el 97 % de su peso en la letra anterior, la 1.3 el 93 % en la de hace dos y la 1.1 el 62 % en la de hace tres: juntas, le pasan al segundo bloque algo parecido al contexto de un 4-grama. En el segundo bloque, la 2.2 pone el 36 % de su peso en la primera letra de la palabra que se está escribiendo, y la 2.4 el 61 % en los saltos de línea, que son apenas el 1.9 % de los caracteres: le sirve para saber cuánto lleva la línea actual y, con eso, cuándo toca el siguiente salto. Nadie programó estas reglas; salieron de minimizar la entropía cruzada. (Otras cabezas reparten el peso de maneras menos fáciles de nombrar; en la frase de la figura, por ejemplo, la 1.4 mira casi siempre al último signo de puntuación.)
#
# Veamos de cerca la cabeza 2.2:

# %%
CABEZA = (1, 1)   # bloque 2, cabeza 2 (índices desde 0): la que mira el inicio de la palabra
fig, ax = plt.subplots(figsize=(9.5, 9))
ax.imshow(pesos[CABEZA][-L:, -L:], cmap=rdp.estilo.SECUENCIAL, vmin=0, vmax=1)
ax.set_xticks(range(L), letras, fontsize=11)
ax.set_yticks(range(L), letras, fontsize=11)
ax.set_xlabel("letra atendida s (clave)")
ax.set_ylabel("letra que consulta t")
ax.grid(False)
ax.set_title(f"Bloque {CABEZA[0] + 1}, cabeza {CABEZA[1] + 1}: pesos α_ts")
plt.show()

# %% [markdown]
# Las franjas verticales están en las primeras letras de las palabras: mientras se escribe «entrambos», las filas de la «n», la «a» o la «o» miran sobre todo a la «e» con que empezó (con la «a», el 70 % del peso). Con esa información, el MLP del bloque puede reconocer qué palabra se está escribiendo y completarla. Cada fila $t$ se usa para predecir la letra $t + 1$, que no está en la figura: la máscara causal deja vacío todo lo que está arriba de la diagonal.

# %% [markdown]
# ## 11. Cierre: un loro estocástico muy pequeño
#
# En 2021, E. M. Bender, T. Gebru, A. McMillan-Major y M. Mitchell (que firmó como «Shmargaret Shmitchell») publicaron «On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?» (*FAccT* 2021). Su argumento central es que un modelo de lenguaje entrenado solo con texto aprende a **reproducir la forma** del lenguaje según cómo se combinan las palabras en sus datos, sin acceso al **significado**, y que la fluidez del texto generado hace que las personas le atribuyan comprensión e intención que no tiene. El artículo también advierte sobre el costo ambiental de entrenar modelos cada vez más grandes y sobre los sesgos de corpus tan enormes que nadie puede documentarlos.
#
# Nuestro *transformer* es un loro estocástico en miniatura: 436,828 parámetros contra los 175,000 millones de GPT-3 (Brown et al., 2020). Nuestro *transformer* es un loro estocástico en miniaturaEscribe «vuesa merced» y abre diálogos con raya porque eso es frecuente en sus datos, no porque sepa quién habla. La tarea de entrenamiento de los modelos grandes es **la misma** que la de esta libreta: predecir el siguiente símbolo. Cuánta comprensión puede emerger de esa tarea a gran escala es una pregunta abierta y muy discutida; lo que esta libreta muestra es que la fluidez local aparece mucho antes que cualquier forma de sentido.
#
# *Referencias:* S. Hochreiter y J. Schmidhuber, «Long short-term memory», *Neural Computation* 9 (1997). D. Bahdanau, K. Cho y Y. Bengio, «Neural machine translation by jointly learning to align and translate», ICLR 2015. A. Vaswani et al., «Attention is all you need», NeurIPS 2017. T. Brown et al., «Language models are few-shot learners», NeurIPS 2020. E. M. Bender, T. Gebru, A. McMillan-Major y S. Shmitchell, «On the dangers of stochastic parrots», FAccT 2021. I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, MIT Press (2016), capítulo 10.

# %% [markdown]
# ## 12. Autoevaluación

# %%
rdp.quiz.mostrar("u06", ids=["u06-p13", "u06-p14", "u06-p15", "u06-p16"])

# %% [markdown]
# ## 13. Ejercicios
#
# 1. **La escala de la atención.** (a) Demuestra que si $q_i, k_i$ son independientes con media 0 y varianza $\sigma^2$, entonces $\operatorname{Var}(\mathbf{q}^\top\mathbf{k}) = d_k\sigma^4$; ¿qué factor de escala habría que usar? (b) Demuestra que la jacobiana del softmax es $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ y que su norma de Frobenius tiende a 0 cuando $\mathbf{p}$ tiende a un vector canónico.
# 2. **Perplejidad.** (a) Demuestra que un modelo que asigna probabilidad $1/k$ a la letra correcta en cada posición tiene perplejidad $k$. (b) Demuestra que la perplejidad del unigrama en los propios datos de entrenamiento es $e^{H(\hat p)}$, con $H(\hat p)$ la entropía de las frecuencias, y compruébalo numéricamente. (c) ¿Por qué la perplejidad de validación de un modelo nunca puede bajar de 1, y qué significaría alcanzarla?
# 3. **La autopista de la LSTM.** Ignorando la dependencia a través de $\mathbf{h}_{t-1}$, muestra que $\partial\mathbf{c}_T/\partial\mathbf{c}_t = \operatorname{diag}\big(\prod_{s=t+1}^{T}\mathbf{f}_s\big)$. Si una unidad tiene $f_s = 0.99$ en todos los pasos, ¿qué fracción del gradiente queda a 100 letras? ¿Y con $f_s = 0.5$? Mide en la LSTM entrenada la distribución de la compuerta de olvido promedio de cada unidad sobre un fragmento de validación (pista: `lstm.lstm.weight_ih_l0` y `weight_hh_l0` apilan las cuatro compuertas en el orden entrada, olvido, candidato, salida).
# 4. **Sin posición.** Demuestra que la autoatención sin máscara es equivariante a permutaciones: si $P$ es una matriz de permutación, $\operatorname{Atención}(PX) = P\operatorname{Atención}(X)$. Luego entrena el *transformer* sin el encaje de posición (con la máscara causal). ¿Qué tanto empeora la perplejidad? ¿Por qué la máscara causal, por sí sola, le da al modelo algo de información sobre la posición?
# 5. **Costo, contexto y presupuesto.** (a) Cuenta las multiplicaciones de una pasada de un bloque de nuestro *transformer* en función de $T$ y $d$ (proyecciones, $QK^\top$, $\alpha V$, MLP). ¿A partir de qué $T$ domina la atención con $d = 128$? (b) Entrena el *transformer* con contexto $T = 32$ y compara la perplejidad, calificando las mismas letras. (c) En Colab con GPU, multiplica por 10 los pasos de las dos redes (el calendario coseno se ajusta solo). ¿Cuánto bajan las perplejidades? ¿Cambia el orden entre la LSTM y el *transformer*?
