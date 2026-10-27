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
# # ¿Es posible aprender? Hoeffding y dimensión VC, con simulaciones
#
# **Unidad 3 · Ejemplo 1**
#
# Un modelo que acierta en todos los ejemplos con los que se entrenó no nos sirve de nada si se equivoca con los ejemplos nuevos. Lo único que podemos medir es el desempeño en los datos que ya vimos; lo que nos importa es el desempeño en los que no hemos visto. En esta libreta nos preguntamos qué relación hay entre los dos y de qué depende. Vamos a:
#
# 1. empezar con una urna de canicas y la **desigualdad de Hoeffding**, que dice cuánto se puede alejar una proporción observada de la verdadera;
# 2. ver que esa desigualdad se aplica a **una** hipótesis fija, pero deja de valer cuando elegimos la mejor de muchas (el experimento de las 1,000 monedas);
# 3. arreglarlo con la **cota de la unión**, y ver por qué es demasiado pesimista;
# 4. contar **dicotomías** para medir la «capacidad» real de un conjunto de hipótesis: función de crecimiento, punto de quiebre y **dimensión VC**;
# 5. comparar la **cota VC** con la brecha real entre el error dentro y fuera de la muestra en un problema simulado.

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
from itertools import product
from math import comb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import linprog

from sklearn.linear_model import LogisticRegression

import rdp

rdp.estilo.usar()
SEMILLA = 2011
rng = np.random.default_rng(SEMILLA)
AZUL, ROJO, DORADO, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS

# %% [markdown]
# ## 1. El problema, y por qué esta vez no hay datos reales
#
# **Problema.** Entrenamos un clasificador con $N$ ejemplos y se equivoca en una fracción $E_{\text{in}}$ de ellos (el **error dentro de la muestra**). Lo que de verdad queremos saber es $E_{\text{out}}$, la probabilidad de que se equivoque con un ejemplo nuevo tomado de la misma población (el **error fuera de la muestra**). Si $E_{\text{in}}$ es pequeño, ¿qué garantía tenemos de que $E_{\text{out}}$ también lo sea? ¿De qué depende esa garantía: del número de ejemplos, del modelo, de cuántos modelos probamos?
#
# **Datos.** Esta libreta no carga ningún archivo: todos los datos los **fabricamos** nosotros. No es por comodidad. Con datos reales nunca conocemos $E_{\text{out}}$ (si lo conociéramos no necesitaríamos teoría); lo más que podemos hacer es estimarlo con un conjunto de prueba, que es otra muestra con su propio error. Cuando simulamos, en cambio, elegimos la población, así que podemos calcular $E_{\text{out}}$ **exactamente** y repetir el experimento miles de veces para ver con qué frecuencia la teoría acierta.
#
# Usaremos la notación de *Learning From Data* (Abu-Mostafa, Magdon-Ismail y Lin, 2012), el libro en el que se basa esta parte del curso:
#
# | Símbolo | Significado |
# |---|---|
# | $f$ | la función objetivo, desconocida, que queremos aprender |
# | $\mathcal{H}$ | el conjunto de hipótesis entre las que elige el algoritmo |
# | $h \in \mathcal{H}$ | una hipótesis cualquiera; $g$ es la que el algoritmo elige al final |
# | $N$ | el número de ejemplos de entrenamiento |
# | $E_{\text{in}}(h)$ | fracción de ejemplos de entrenamiento en los que $h$ se equivoca |
# | $E_{\text{out}}(h)$ | probabilidad de que $h$ se equivoque en un punto nuevo |

# %% [markdown]
# ## 2. La urna y la desigualdad de Hoeffding
#
# Olvidemos por un momento el aprendizaje. Tenemos una urna con muchísimas canicas, rojas y verdes. Una fracción $\mu$ es roja, pero **no sabemos** cuánto vale $\mu$. Sacamos $N$ canicas al azar, de forma independiente, y observamos la fracción $\nu$ de rojas en la muestra. ¿Qué nos dice $\nu$ sobre $\mu$?
#
# No puede decirnos el valor exacto: con mala suerte, una urna mitad roja podría darnos 10 canicas verdes seguidas. Pero intuimos que, con $N$ grande, $\nu$ casi siempre estará cerca de $\mu$. Veámoslo con una urna en la que $\mu = 0.4$ (lo sabemos porque nosotros la llenamos):

# %%
mu = 0.4
for N in (10, 100, 1000):
    nus = rng.binomial(N, mu, size=6) / N
    print(f"N = {N:>4}: seis muestras dan ν =", "  ".join(f"{v:.3f}" for v in nus))

# %% [markdown]
# Con $N = 10$, $\nu$ brinca entre valores muy lejanos de 0.4; con $N = 1000$ casi no se mueve. La **desigualdad de Hoeffding** convierte esa intuición en un número.
#
# > **Desigualdad de Hoeffding.** Sean $X_1, \dots, X_N$ variables aleatorias **independientes** con valores en $\{0, 1\}$ y la misma media $\mu$, y sea $\nu = \frac{1}{N}\sum_{i=1}^N X_i$. Para todo $\varepsilon > 0$,
# > $$
# > \mathbb{P}\big[\,|\nu - \mu| > \varepsilon\,\big] \;\le\; 2\,e^{-2\varepsilon^2 N}.
# > $$
#
# Lo notable es lo que **no** aparece en la cota: no depende de $\mu$ ni del tamaño de la urna, solo de $N$ y de la tolerancia $\varepsilon$. Hay dos hipótesis que no se deben olvidar, porque todo lo que sigue depende de ellas:
#
# 1. **Las extracciones son independientes** y vienen de la misma urna. (Si muestreamos solo de la parte de arriba de la urna, no hay garantía.)
# 2. **$\mu$ está fija antes de muestrear.** La cota habla de una urna concreta; no dice nada si, después de ver las muestras, elegimos la urna que más nos conviene. Esta segunda hipótesis será la que se rompa en la sección 4.
#
# **Esbozo de la demostración.** Basta acotar una cola, $\mathbb{P}[\nu - \mu > \varepsilon]$; la otra es simétrica y por eso aparece el 2. Para cualquier $s > 0$, como $e^{sx}$ es creciente, la desigualdad de Markov da
# $$
# \mathbb{P}[\nu - \mu > \varepsilon] = \mathbb{P}\!\left[e^{sN(\nu-\mu)} > e^{sN\varepsilon}\right] \le e^{-sN\varepsilon}\,\mathbb{E}\!\left[e^{sN(\nu-\mu)}\right] = e^{-sN\varepsilon}\prod_{i=1}^N \mathbb{E}\!\left[e^{s(X_i-\mu)}\right],
# $$
# donde el producto sale de la independencia. El **lema de Hoeffding** dice que si $X \in [0, 1]$, entonces $\mathbb{E}[e^{s(X - \mu)}] \le e^{s^2/8}$. Así, $\mathbb{P}[\nu-\mu>\varepsilon] \le e^{-sN\varepsilon + s^2 N/8}$ para todo $s>0$; el exponente es mínimo en $s = 4\varepsilon$, y ahí vale $-2\varepsilon^2 N$.
#
# Comprobémoslo. Para cada $N$ simulamos un millón de muestras, contamos qué fracción cae a más de $\varepsilon$ de $\mu$ y lo comparamos con la cota. Como $N\nu$ es binomial, también podemos calcular la probabilidad exacta, que nos sirve para los casos en que el evento es tan raro que un millón de simulaciones no alcanza a verlo.

# %%
def prob_exacta(N, mu, eps):
    """P[|ν − μ| > ε] cuando Nν ~ Binomial(N, μ)."""
    k = np.arange(N + 1)
    return binom.pmf(k, N, mu)[np.abs(k / N - mu) > eps + 1e-12].sum()


def hoeffding(N, eps):
    return 2 * np.exp(-2 * eps**2 * N)


REPETICIONES = 1_000_000
filas = []
for N in (10, 50, 100, 200, 1000):
    nus = rng.binomial(N, mu, size=REPETICIONES) / N
    for eps in (0.05, 0.1, 0.2):
        filas.append({"N": N, "ε": eps,
                      "frecuencia simulada": np.mean(np.abs(nus - mu) > eps + 1e-12),
                      "probabilidad exacta": prob_exacta(N, mu, eps),
                      "cota de Hoeffding": hoeffding(N, eps)})
urna = pd.DataFrame(filas)
urna.style.format({"ε": "{:.2f}", "frecuencia simulada": "{:.2e}", "probabilidad exacta": "{:.2e}", "cota de Hoeffding": "{:.2e}"}).hide(axis="index")

# %% [markdown]
# Tres cosas que leer en la tabla:
#
# - **La cota nunca se viola.** En todos los renglones la frecuencia observada está por debajo de la cota.
# - **Las simulaciones no ven los eventos muy raros.** Con $N = 1000$ y $\varepsilon = 0.1$, ninguna de un millón de muestras se alejó tanto; la probabilidad exacta es del orden de $10^{-10}$.
- **Con $N$ pequeño la cota no dice nada.** Para $N = 10$ y $\varepsilon = 0.05$ la cota es mayor que 1, y una probabilidad siempre es menor que 1.
# - **La cota es holgada, pero tiene la forma correcta.** Para $N = 200$ y $\varepsilon = 0.1$ la probabilidad real es unas 12 veces menor que la cota; aun así, las dos caen exponencialmente con $N$. Esto se ve mejor en escala logarítmica:

# %%
epsilons = np.linspace(0.005, 0.3, 120)
fig, ax = plt.subplots(figsize=(9, 5.5))
for N, color in zip((10, 50, 200), (GRIS, DORADO, AZUL)):
    exacta = np.array([prob_exacta(N, mu, e) for e in epsilons])
    ax.plot(epsilons, exacta, color=color, drawstyle="steps-post")
    ax.plot(epsilons, hoeffding(N, epsilons), color=color, linestyle="--", linewidth=1.5)
ax.set_yscale("log")
ax.set(ylim=(1e-9, 3), xlim=(0, 0.34),
       title="Probabilidad de que la muestra se aleje más de ε de la urna (μ = 0.4)",
       xlabel="tolerancia ε", ylabel="P[ |ν − μ| > ε ]  (escala log)")
for x, y, N, color in ((0.305, 0.3, 10, GRIS), (0.305, 2e-4, 50, DORADO), (0.235, 2e-8, 200, AZUL)):
    ax.text(x, y, f"N = {N}", va="center", fontsize=11.5, color=color, fontweight="semibold")
ax.axhline(1, color=rdp.estilo.TENUE, linewidth=0.8)
ax.text(0.25, 3e-7, "continua: probabilidad exacta\ndiscontinua: cota de Hoeffding",
        fontsize=10.5, color=rdp.estilo.TENUE, va="center")
plt.show()

# %% [markdown]
# En escala logarítmica, la cota $2e^{-2\varepsilon^2 N}$ es una parábola invertida, y la probabilidad real cae con una forma parecida, siempre por debajo. Cuadruplicar $N$ tiene el mismo efecto en la cota que duplicar $\varepsilon$: la precisión mejora como $1/\sqrt{N}$.
#
# Otra forma útil de leer la desigualdad: si fijamos la probabilidad de fallar en $\delta$ y despejamos $\varepsilon$, obtenemos que **con probabilidad al menos $1 - \delta$**,
# $$
# |\nu - \mu| \le \sqrt{\frac{1}{2N}\ln\frac{2}{\delta}}.
# $$
# Con $N = 1000$ y $\delta = 0.05$ eso da $|\nu - \mu| \le 0.043$: una «encuesta» de mil canicas fija la proporción con un margen de unos 4 puntos porcentuales, sea cual sea la urna.

# %% [markdown]
# ## 3. De la urna al aprendizaje: una hipótesis fija
#
# ¿Qué tiene que ver la urna con aprender? Tomemos un problema de clasificación y **una** hipótesis $h$ fija, elegida antes de ver los datos. Cada punto $\mathbf{x}$ del espacio es una canica: es «roja» si $h$ se equivoca en él ($h(\mathbf{x}) \ne y$) y «verde» si acierta. Entonces:
#
# - la fracción de canicas rojas en la urna, $\mu$, es la probabilidad de que $h$ se equivoque en un punto nuevo: **$E_{\text{out}}(h)$**;
# - la fracción de rojas en la muestra, $\nu$, es la fracción de ejemplos de entrenamiento en que se equivoca: **$E_{\text{in}}(h)$**.
#
# Si los $N$ ejemplos son independientes, Hoeffding se aplica tal cual:
# $$
# \mathbb{P}\big[\,|E_{\text{in}}(h) - E_{\text{out}}(h)| > \varepsilon\,\big] \le 2e^{-2\varepsilon^2 N}.
# $$
#
# **El problema sintético.** Los puntos $\mathbf{x} = (x_1, x_2)$ se distribuyen uniformemente en el cuadrado $[-1, 1]^2$. La función objetivo es una recta, $f(\mathbf{x}) = \operatorname{sign}(x_2 - 0.6\,x_1 - 0.2)$, y para que el problema no sea demasiado limpio, cada etiqueta se invierte al azar con probabilidad $\rho = 0.1$ (ruido, como el de un etiquetador que a veces se equivoca). Como todo es lineal y uniforme, el error de cualquier clasificador lineal $h(\mathbf{x}) = \operatorname{sign}(\mathbf{w}\cdot\mathbf{x} + b)$ se puede calcular **exactamente**: la probabilidad de que $h$ y $f$ no coincidan es el área de la región entre las dos rectas, dividida entre 4 (el área del cuadrado), y entonces
# $$
# E_{\text{out}}(h) = \mathbb{P}[h(\mathbf{x}) \ne y] = \rho + (1 - 2\rho)\,\mathbb{P}[h(\mathbf{x}) \ne f(\mathbf{x})],
# $$
# porque $h$ se equivoca si discrepa de $f$ y la etiqueta no se invirtió, o si coincide con $f$ y la etiqueta sí se invirtió. Las áreas las calculamos recortando el cuadrado con semiplanos (el recorte de un polígono convexo por un semiplano es otro polígono convexo) y aplicando la fórmula del agujeta (*shoelace*).

# %%
CUADRADO = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
W_F, B_F = np.array([-0.6, 1.0]), -0.2   # función objetivo: x2 − 0.6 x1 − 0.2 > 0
RUIDO = 0.1


def recortar(poligono, w, b):
    """Parte del polígono convexo donde w·x + b ≥ 0 (algoritmo de Sutherland–Hodgman)."""
    salida = []
    for p, q in zip(poligono, np.roll(poligono, -1, axis=0)):
        fp, fq = w @ p + b, w @ q + b
        if fp >= 0:
            salida.append(p)
        if fp * fq < 0:
            salida.append(p + fp / (fp - fq) * (q - p))
    return np.array(salida)


def area(poligono):
    if len(poligono) < 3:
        return 0.0
    x, y = poligono[:, 0], poligono[:, 1]
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def e_out(w, b):
    """Error fuera de la muestra exacto de sign(w·x + b) con x uniforme en [−1, 1]² y ruido RUIDO."""
    discrepa = (area(recortar(recortar(CUADRADO, W_F, B_F), -w, -b))
                + area(recortar(recortar(CUADRADO, -W_F, -B_F), w, b))) / 4
    return RUIDO + (1 - 2 * RUIDO) * discrepa


def muestra(N, rng):
    X = rng.uniform(-1, 1, size=(N, 2))
    y = np.where(X @ W_F + B_F > 0, 1, -1)
    y[rng.random(N) < RUIDO] *= -1
    return X, y


def e_in(w, b, X, y):
    return np.mean(np.where(X @ w + b > 0, 1, -1) != y)


print(f"E_out de la propia f (solo el ruido): {e_out(W_F, B_F):.4f}")

# %% [markdown]
# La función objetivo misma se equivoca 10 % de las veces, por el ruido: ningún clasificador puede bajar de ahí. Ahora fijamos una hipótesis razonable pero imperfecta, la recta horizontal $h(\mathbf{x}) = \operatorname{sign}(x_2 - 0.1)$, y la evaluamos en 20,000 muestras distintas de $N = 100$ puntos.

# %%
W_H, B_H = np.array([0.0, 1.0]), -0.1
eout_h = e_out(W_H, B_H)

N, R = 100, 20_000
eins_h = np.array([e_in(W_H, B_H, *muestra(N, rng)) for _ in range(R)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1, 1.4]})
Xm, ym = muestra(200, np.random.default_rng(SEMILLA))
for etiqueta, color in ((1, AZUL), (-1, ROJO)):
    ax1.scatter(*Xm[ym == etiqueta].T, s=18, color=color, alpha=0.8, edgecolors="none")
xs = np.array([-1, 1])
ax1.plot(xs, 0.6 * xs + 0.2, color=rdp.estilo.TINTA, linewidth=2)
ax1.plot(xs, [0.1, 0.1], color=DORADO, linewidth=2.5, linestyle="--")
ax1.text(-0.95, 0.75, "objetivo f", color=rdp.estilo.TINTA, fontsize=11.5, fontweight="semibold")
ax1.text(0.35, -0.02, "hipótesis h", color=DORADO, fontsize=11.5, fontweight="semibold", va="top")
ax1.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", title="200 puntos del problema (10 % de ruido)",
        xlabel="$x_1$", ylabel="$x_2$")

ax2.hist(eins_h, bins=np.arange(0, 0.5, 0.01) - 0.005, color=AZUL, alpha=0.85)
ax2.axvline(eout_h, color=rdp.estilo.TINTA, linewidth=2)
ax2.text(eout_h + 0.005, ax2.get_ylim()[1] * 0.93, f"$E_{{out}}(h) = {eout_h:.3f}$", fontsize=12)
ax2.set(title=f"$E_{{in}}(h)$ en {R:,} muestras de N = {N}", xlabel="$E_{in}(h)$", ylabel="número de muestras")
plt.tight_layout()
plt.show()

for eps in (0.05, 0.1, 0.15):
    print(f"ε = {eps:.2f}:  frecuencia de |E_in − E_out| > ε = {np.mean(np.abs(eins_h - eout_h) > eps + 1e-12):.4f}"
          f"   cota de Hoeffding = {hoeffding(N, eps):.4f}")

# %% [markdown]
# $E_{\text{in}}(h)$ es una variable aleatoria (depende de qué puntos nos tocaron) centrada en $E_{\text{out}}(h)$, y Hoeffding acota bien sus colas. Hasta aquí todo va bien, pero **esto no es aprendizaje**: la hipótesis la elegimos nosotros antes de ver los datos. Lo que dice Hoeffding con una sola $h$ es que podemos **verificar** una hipótesis, no que podamos **encontrar** una buena.

# %% [markdown]
# ## 4. El problema de elegir: el experimento de las 1,000 monedas
#
# Aprender es elegir, entre muchas hipótesis, la que mejor se ve en los datos. Veamos qué le pasa a Hoeffding cuando hacemos eso con un experimento que no tiene nada que aprender (es el ejercicio 1.10 de *Learning From Data*):
#
# - Tomamos 1,000 monedas **legales** ($\mu = 0.5$ para todas) y lanzamos cada una 10 veces.
# - $c_1$ es la primera moneda; $c_{\text{rand}}$ es una moneda elegida al azar; $c_{\min}$ es la moneda que sacó **menos águilas** (si hay empate, la primera de ellas).
# - Para cada una registramos su fracción de águilas: $\nu_1$, $\nu_{\text{rand}}$ y $\nu_{\min}$.
#
# Las tres monedas son legales, así que para las tres $\mu = 0.5$. Repetimos el experimento 100,000 veces.

# %%
MONEDAS, LANZAMIENTOS, EXPERIMENTOS = 1000, 10, 100_000
nu_1, nu_rand, nu_min = [], [], []
for _ in range(EXPERIMENTOS // 10_000):  # en bloques, para no usar demasiada memoria
    aguilas = rng.binomial(LANZAMIENTOS, 0.5, size=(10_000, MONEDAS)).astype(np.int8)
    nu_1.append(aguilas[:, 0])
    nu_rand.append(aguilas[np.arange(10_000), rng.integers(MONEDAS, size=10_000)])
    nu_min.append(aguilas.min(axis=1))
monedas = {nombre: np.concatenate(v) / LANZAMIENTOS
           for nombre, v in (("ν_1", nu_1), ("ν_rand", nu_rand), ("ν_min", nu_min))}
EN_LATEX = {"ν_1": "$\\nu_1$", "ν_rand": "$\\nu_{\\mathrm{rand}}$", "ν_min": "$\\nu_{\\min}$"}

fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (nombre, v), color in zip(axs, monedas.items(), (AZUL, DORADO, ROJO)):
    valores = np.arange(11) / 10
    ax.bar(valores, [np.mean(np.isclose(v, x)) for x in valores], width=0.08, color=color)
    ax.axvline(0.5, color=rdp.estilo.TINTA, linewidth=1, linestyle=":")
    ax.set(title=f"{EN_LATEX[nombre]}  (promedio {v.mean():.3f})", xlabel="fracción de águilas", xlim=(-0.06, 1.06))
axs[0].set_ylabel("proporción de experimentos")
fig.suptitle("Tres monedas legales, 10 lanzamientos cada una, 100,000 experimentos", y=1.02)
plt.show()

# %% [markdown]
# $\nu_1$ y $\nu_{\text{rand}}$ tienen la distribución binomial que esperábamos, centrada en 0.5. $\nu_{\min}$ no: casi siempre vale 0 o 0.1. Entre 1,000 monedas legales, es casi seguro que alguna saque 0 o 1 águila en 10 lanzamientos (la probabilidad de que una moneda dada saque a lo más una es $11/1024 \approx 1\%$, y tenemos 1,000 oportunidades). Comparemos con Hoeffding:

# %%
eps_m = np.linspace(0, 0.5, 51)
fig, ax = plt.subplots(figsize=(9, 5.5))
ultimo = {}
for (nombre, v), color in zip(monedas.items(), (AZUL, DORADO, ROJO)):
    frec = [np.mean(np.abs(v - 0.5) > e + 1e-12) for e in eps_m]
    ax.plot(eps_m, frec, color=color, drawstyle="steps-post", linewidth=2.4 if nombre == "ν_min" else 2)
ax.plot(eps_m, hoeffding(LANZAMIENTOS, eps_m), color=rdp.estilo.TINTA, linestyle="--", linewidth=1.8)
ax.text(0.41, 0.7, "$\\nu_{\\min}$", color=ROJO, fontsize=14)
ax.text(0.29, 0.2, "$\\nu_1$ y $\\nu_{\\mathrm{rand}}$ (casi encimadas)", color=AZUL, fontsize=11.5)
ax.annotate("cota de Hoeffding  $2e^{-2\\varepsilon^2 N}$", xy=(0.15, 1.27), xytext=(0.22, 1.55),
            color=rdp.estilo.TINTA, fontsize=11.5, arrowprops={"arrowstyle": "-", "color": rdp.estilo.TENUE})
ax.set(title="Frecuencia de |ν − μ| > ε frente a la cota de Hoeffding (N = 10)",
       xlabel="tolerancia ε", ylabel="frecuencia de |ν − 0.5| > ε", ylim=(0, 2.05))
plt.show()

tabla_monedas = pd.DataFrame({nombre: [np.mean(np.abs(v - 0.5) > e + 1e-12) for e in (0.2, 0.3, 0.4)]
                              for nombre, v in monedas.items()}, index=[0.2, 0.3, 0.4])
tabla_monedas["cota de Hoeffding"] = hoeffding(LANZAMIENTOS, tabla_monedas.index.to_numpy())
tabla_monedas.index.name = "ε"
tabla_monedas.round(4)

# %% [markdown]
# Para $c_1$ y $c_{\text{rand}}$ la cota se cumple. Para $c_{\min}$ se viola de manera escandalosa: con $\varepsilon = 0.4$, Hoeffding promete que $|\nu - \mu| > 0.4$ ocurre a lo más el 8 % de las veces, y para $c_{\min}$ ocurre en más de la mitad de los experimentos.
#
# ¿Contradice esto a Hoeffding? No: viola su **segunda hipótesis**. $c_1$ y $c_{\text{rand}}$ se eligieron sin mirar los lanzamientos; $c_{\min}$ se eligió **después** de verlos, justamente porque su muestra era extrema.
#
# **La moraleja para el aprendizaje.** Cada moneda es una hipótesis; las águilas son errores; las 1,000 monedas son $\mathcal{H}$ y elegir $c_{\min}$ es exactamente lo que hace un algoritmo de aprendizaje: quedarse con la hipótesis $g$ de menor $E_{\text{in}}$. Si $\mathcal{H}$ es grande, siempre habrá alguna hipótesis con $E_{\text{in}}$ pequeño **por pura suerte**, y su $E_{\text{in}}$ no dice nada de su $E_{\text{out}}$. Una moneda que sacó 0 águilas en 10 tiros no es una moneda que nunca saque águila.

# %% [markdown]
# ## 5. La cota de la unión
#
# Supongamos que $\mathcal{H} = \{h_1, \dots, h_M\}$ es finito. No sabemos de antemano cuál será $g$, pero sí sabemos que es alguna de las $h_m$. Si $g$ tiene una brecha mayor que $\varepsilon$, entonces **alguna** $h_m$ la tiene:
# $$
# \big\{|E_{\text{in}}(g) - E_{\text{out}}(g)| > \varepsilon\big\} \subseteq \bigcup_{m=1}^{M} \big\{|E_{\text{in}}(h_m) - E_{\text{out}}(h_m)| > \varepsilon\big\}.
# $$
# La probabilidad de una unión es a lo más la suma de las probabilidades, y a cada sumando, que corresponde a una hipótesis fija, sí le podemos aplicar Hoeffding:
# $$
# \mathbb{P}\big[|E_{\text{in}}(g) - E_{\text{out}}(g)| > \varepsilon\big] \le \sum_{m=1}^{M} \mathbb{P}\big[|E_{\text{in}}(h_m) - E_{\text{out}}(h_m)| > \varepsilon\big] \le 2M e^{-2\varepsilon^2 N}.
# $$
# Esta cota ya no depende de cómo elijamos $g$: vale para cualquier algoritmo. El precio es el factor $M$. Despejando como antes, con probabilidad al menos $1 - \delta$,
# $$
# E_{\text{out}}(g) \le E_{\text{in}}(g) + \sqrt{\frac{1}{2N}\ln\frac{2M}{\delta}}.
# $$
# El margen crece solo como $\sqrt{\ln M}$: un conjunto de hipótesis mil veces mayor se paga con algunas decenas más de datos, no con mil veces más. Pero aparecen dos problemas.
#
# **Primero, la cota es muy pesimista.** La suma de probabilidades solo es igual a la probabilidad de la unión cuando los eventos no se traslapan. Con las monedas, que son independientes, el traslape es pequeño. Pero las hipótesis de un modelo real se parecen mucho entre sí: dos rectas casi iguales se equivocan casi en los mismos puntos, así que si una tiene mala suerte con la muestra, la otra también; sus eventos «malos» se traslapan casi por completo, y sumarlos cuenta lo mismo muchas veces.
#
# Lo vemos comparando dos conjuntos de $M$ hipótesis con $N = 100$ y $\varepsilon = 0.1$:
#
# - **independientes:** $M$ monedas legales, cada una con su propia muestra;
# - **parecidas:** $M$ «rayos» $h_a(x) = \operatorname{sign}(x - a)$ en $[0, 1]$, con los umbrales $a$ repartidos uniformemente y evaluados en la **misma** muestra, para el objetivo $f(x) = \operatorname{sign}(x - 0.5)$ sin ruido. Aquí $E_{\text{out}}(h_a) = |a - 0.5|$ y $E_{\text{in}}(h_a)$ es la fracción de puntos de la muestra que caen entre $a$ y $0.5$.
#
# Para cada $M$ estimamos la probabilidad de que **al menos una** de las $M$ hipótesis tenga una brecha mayor que $\varepsilon$, que es lo que la cota de la unión intenta acotar.

# %%
N_U, EPS_U, R_U = 100, 0.1, 4000
Ms = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])

# monedas independientes: P[alguna se aleja] = 1 − (1 − p)^M, con p exacta de una sola
p_una = prob_exacta(N_U, 0.5, EPS_U)
indep = 1 - (1 - p_una) ** Ms

# rayos parecidos, evaluados todos en la misma muestra
parecidas = []
muestras = np.sort(rng.uniform(0, 1, size=(R_U, N_U)), axis=1)
antes_de_medio = (muestras < 0.5).sum(axis=1)
for M in Ms:
    a = np.linspace(0, 1, M + 2)[1:-1] if M > 1 else np.array([0.3])
    antes_de_a = np.stack([np.searchsorted(fila, a) for fila in muestras])     # R_U × M
    brecha = np.abs(np.abs(antes_de_a - antes_de_medio[:, None]) / N_U - np.abs(a - 0.5))
    parecidas.append(np.mean((brecha > EPS_U + 1e-12).any(axis=1)))
parecidas = np.array(parecidas)
union = np.minimum(2 * Ms * np.exp(-2 * EPS_U**2 * N_U), 1e3)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(Ms, union, color=rdp.estilo.TINTA, linestyle="--", marker="o", markersize=4)
ax.plot(Ms, indep, color=ROJO, marker="o")
ax.plot(Ms, parecidas, color=AZUL, marker="o")
ax.axhline(1, color=rdp.estilo.TENUE, linewidth=0.8)
ax.set(xscale="log", yscale="log", ylim=(5e-3, 2e2), xlim=(0.8, 3000),
       title=f"Probabilidad de que alguna de M hipótesis se aleje más de ε = {EPS_U} (N = {N_U})",
       xlabel="número de hipótesis M (escala log)", ylabel="probabilidad (escala log)")
ax.text(2.5, 25, "cota de la unión  $2Me^{-2\\varepsilon^2 N}$", fontsize=11.5, color=rdp.estilo.TINTA)
for nombre, y, color in (("independientes", indep[-1], ROJO), ("parecidas (rayos)", parecidas[-1], AZUL)):
    ax.text(1150, y, nombre, va="center", fontsize=11, color=color)
plt.show()

# %% [markdown]
# Con hipótesis independientes, la probabilidad real crece casi en proporción a $M$, como la cota (aunque por debajo de ella), hasta saturarse en 1. Con hipótesis parecidas, la probabilidad de que **alguna** se aleje apenas crece: pasar de 10 a 1,000 rayos, cien veces más hipótesis, solo la lleva de 0.12 a 0.25, porque los rayos con umbrales vecinos se equivocan en los mismos puntos. La cota de la unión, en cambio, sigue creciendo como $M$.
#
# **Segundo, $M$ suele ser infinito.** Un perceptrón en el plano tiene una recta por cada valor de $(w_1, w_2, b)$: una cantidad infinita de hipótesis. Con $M = \infty$ la cota dice $\mathbb{P}[\dots] \le \infty$, que no sirve. La gráfica sugiere la salida: lo que importa no es cuántas hipótesis hay, sino **cuántas se comportan de manera distinta** sobre los datos.

# %% [markdown]
# ## 6. Dicotomías: contar hipótesis efectivas
#
# Dados $N$ puntos $\mathbf{x}_1, \dots, \mathbf{x}_N$, una hipótesis los etiqueta con un vector $(h(\mathbf{x}_1), \dots, h(\mathbf{x}_N)) \in \{-1, +1\}^N$. A cada vector distinto que $\mathcal{H}$ puede producir se le llama **dicotomía**. Infinitas rectas producen solo un número finito de dicotomías sobre $N$ puntos, a lo más $2^N$: desde el punto de vista de la muestra, dos hipótesis con la misma dicotomía son la misma.
#
# > **Función de crecimiento.** $m_{\mathcal{H}}(N)$ es el **máximo** número de dicotomías que $\mathcal{H}$ puede generar sobre $N$ puntos:
# > $$
# > m_{\mathcal{H}}(N) = \max_{\mathbf{x}_1, \dots, \mathbf{x}_N} \big|\{(h(\mathbf{x}_1), \dots, h(\mathbf{x}_N)) : h \in \mathcal{H}\}\big| \;\le\; 2^N.
# > $$
#
# Si $\mathcal{H}$ genera las $2^N$ dicotomías en algún conjunto de $N$ puntos, se dice que **rompe** (*shatters*) esos puntos. El primer $N$ en que eso ya no es posible para ningún conjunto de puntos es el **punto de quiebre** $k$, y la **dimensión VC** es el mayor $N$ que sí se puede romper: $d_{\text{VC}} = k - 1$.
#
# Vamos a contar dicotomías por computadora para tres conjuntos de hipótesis:
#
# | Conjunto | Hipótesis | Fórmula conocida |
# |---|---|---|
# | rayos positivos en $\mathbb{R}$ | $h(x) = \operatorname{sign}(x - a)$ | $m_{\mathcal{H}}(N) = N + 1$ |
# | intervalos positivos en $\mathbb{R}$ | $h(x) = +1$ si $a < x < b$, si no $-1$ | $m_{\mathcal{H}}(N) = \tfrac{1}{2}N^2 + \tfrac{1}{2}N + 1$ |
# | perceptrones en $\mathbb{R}^2$ | $h(\mathbf{x}) = \operatorname{sign}(w_1 x_1 + w_2 x_2 + b)$ | (lo descubriremos) |
#
# **Rayos e intervalos.** En la recta, solo importa el orden de los puntos, así que podemos usar $x = 1, 2, \dots, N$. Generamos muchísimas hipótesis al azar y contamos cuántos vectores de etiquetas distintos aparecen. (Con hipótesis al azar podríamos no encontrar alguna dicotomía rara, pero con cientos de miles de sorteos y huecos de ancho 1 es prácticamente imposible.)

# %%
def dicotomias_rayos(N, rng, sorteos=50_000):
    x = np.arange(1, N + 1)
    a = rng.uniform(0, N + 1, size=sorteos)
    return len(np.unique(x[None, :] > a[:, None], axis=0))


def dicotomias_intervalos(N, rng, sorteos=200_000):
    x = np.arange(1, N + 1)
    extremos = np.sort(rng.uniform(0, N + 1, size=(sorteos, 2)), axis=1)
    return len(np.unique((x > extremos[:, :1]) & (x < extremos[:, 1:]), axis=0))


Ns = np.arange(1, 11)
rayos = [dicotomias_rayos(N, rng) for N in Ns]
intervalos = [dicotomias_intervalos(N, rng) for N in Ns]
print("rayos:     ", rayos, " ¿igual a N + 1?", rayos == list(Ns + 1))
print("intervalos:", intervalos, " ¿igual a N²/2 + N/2 + 1?", intervalos == list(Ns * (Ns + 1) // 2 + 1))

# %% [markdown]
# Las cuentas coinciden con las fórmulas. La de los rayos es fácil de ver: los $N$ puntos dejan $N + 1$ huecos donde puede caer $a$, y cada hueco da una dicotomía. Para los intervalos, los dos extremos caen en dos de los $N + 1$ huecos distintos, $\binom{N+1}{2}$ maneras, más la dicotomía «todos $-1$» cuando caen en el mismo hueco. Los rayos rompen 1 punto pero no 2 (no pueden etiquetar $(+1, -1)$), así que $k = 2$ y $d_{\text{VC}} = 1$; los intervalos rompen 2 pero no 3 (no pueden hacer $(+1, -1, +1)$), así que $k = 3$ y $d_{\text{VC}} = 2$.
#
# **Perceptrones en el plano.** Aquí no conviene sortear rectas: hay dicotomías que solo se logran con rectas muy particulares. Mejor revisamos **cada uno** de los $2^N$ vectores de etiquetas y preguntamos si alguna recta lo produce. Eso es un problema de **factibilidad lineal**: ¿existen $(w_1, w_2, b)$ con
# $$
# y_i\,(w_1 x_{i1} + w_2 x_{i2} + b) \ge 1 \quad \text{para } i = 1, \dots, N?
# $$
# (Pedir $\ge 1$ en vez de $> 0$ no pierde generalidad: si hay una solución con margen positivo, se reescala.) Para decidir basta un programa lineal con objetivo cero; nosotros minimizamos además $|w_1| + |w_2|$, lo que no cambia la respuesta pero da rectas con más margen, más fáciles de ver en las figuras. `scipy.optimize.linprog` lo resuelve en milisegundos. Tomamos puntos al azar, que con probabilidad 1 están en **posición general** (no hay tres alineados).

# %%
def separable(X, y):
    """¿Hay una recta que produzca las etiquetas y sobre los puntos X? Devuelve (w1, w2, b) o None."""
    n = len(X)
    # variables (w1, w2, b, t1, t2); restricciones y_i (w·x_i + b) ≥ 1 y |w_j| ≤ t_j
    A = np.zeros((n + 4, 5))
    A[:n, :3] = -y[:, None] * np.column_stack([X, np.ones(n)])
    A[n:, :2] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    A[n:, 3:] = -np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    cotas = np.concatenate([-np.ones(n), np.zeros(4)])
    # minimizar t1 + t2 no es necesario para decidir, pero da rectas más centradas para dibujarlas
    res = linprog([0, 0, 0, 1, 1], A_ub=A, b_ub=cotas, bounds=[(None, None)] * 3 + [(0, None)] * 2, method="highs")
    return res.x[:3] if res.status == 0 else None


def dicotomias_perceptron(X):
    return [(np.array(y), separable(X, np.array(y))) for y in product((-1, 1), repeat=len(X))]


puntos = {N: rng.uniform(-1, 1, size=(N, 2)) for N in range(1, 7)}
perceptron = [sum(w is not None for _, w in dicotomias_perceptron(puntos[N])) for N in range(1, 7)]
print("perceptrón en R²:", perceptron)

# %% [markdown]
# Hasta $N = 3$ obtenemos $2^N$: el perceptrón rompe 3 puntos. En $N = 4$ obtenemos 14 de 16. Dibujemos las dicotomías para verlo. Con 3 puntos en posición general, todas las etiquetas se separan:

# %%
def dibujar_dicotomias(X, columnas, titulo):
    resultados = dicotomias_perceptron(X)
    filas_fig = int(np.ceil(len(resultados) / columnas))
    fig, axs = plt.subplots(filas_fig, columnas, figsize=(2.6 * columnas, 2.6 * filas_fig))
    t = np.linspace(-1.5, 1.5, 2)
    for ax, (y, w) in zip(axs.ravel(), resultados):
        if w is not None and np.abs(w[:2]).max() > 1e-9:  # las dicotomías constantes no necesitan recta
            if abs(w[1]) > abs(w[0]):
                ax.plot(t, -(w[0] * t + w[2]) / w[1], color=rdp.estilo.TENUE, linewidth=1.5)
            else:
                ax.plot(-(w[1] * t + w[2]) / w[0], t, color=rdp.estilo.TENUE, linewidth=1.5)
        elif w is None:
            ax.set_facecolor("#F6E3E5")
            ax.text(0, -1.15, "imposible", ha="center", fontsize=11, color=ROJO, fontweight="semibold")
        ax.scatter(*X[y == 1].T, s=90, color=AZUL, zorder=3)
        ax.scatter(*X[y == -1].T, s=90, color=ROJO, marker="s", zorder=3)
        ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), xticks=[], yticks=[], aspect="equal")
        ax.grid(False)
    fig.suptitle(titulo, y=1.0)
    plt.tight_layout()
    plt.show()


tres = np.array([[-0.8, -0.6], [0.8, -0.5], [0.0, 0.8]])
dibujar_dicotomias(tres, 4, "Las 8 dicotomías de 3 puntos: una recta logra todas (círculo azul = +1, cuadro rojo = −1)")

# %% [markdown]
# Con 4 puntos hay 2 dicotomías imposibles. Si los puntos forman un cuadrilátero convexo, son las dos «diagonales» (el patrón del o exclusivo, XOR): ninguna recta deja dos vértices opuestos de un lado y los otros dos del otro.

# %%
cuatro = np.array([[-0.8, -0.7], [0.7, -0.8], [0.8, 0.7], [-0.7, 0.8]])
dibujar_dicotomias(cuatro, 4, "Las 16 dicotomías de 4 puntos: dos no se pueden lograr con una recta")

# %% [markdown]
# Esto no prueba que **ningún** conjunto de 4 puntos se pueda romper; para eso hay que considerar también el otro caso de posición general, un punto dentro del triángulo que forman los otros tres. Ahí la dicotomía imposible es «el punto interior contra los tres vértices» (y su opuesta): cualquier semiplano que contenga a los tres vértices contiene su envolvente convexa. Y si tres puntos están alineados, el del medio no puede llevar etiqueta distinta a la de los extremos. Así que en todos los casos faltan dicotomías: **el punto de quiebre del perceptrón en $\mathbb{R}^2$ es $k = 4$ y su dimensión VC es $d_{\text{VC}} = 3$**, el número de parámetros $(w_1, w_2, b)$. (En general, el perceptrón en $\mathbb{R}^d$ tiene $d_{\text{VC}} = d + 1$.)
#
# Juntemos todas las cuentas:

# %%
def sauer(N, k):
    return sum(comb(N, i) for i in range(k))


tabla_m = pd.DataFrame({
    "N": Ns, "2^N": 2**Ns,
    "rayos": rayos, "intervalos": intervalos,
    "perceptrón R²": perceptron + [np.nan] * (len(Ns) - len(perceptron)),
    "Sauer k=2": [sauer(N, 2) for N in Ns],
    "Sauer k=3": [sauer(N, 3) for N in Ns],
    "Sauer k=4": [sauer(N, 4) for N in Ns],
}).set_index("N")
tabla_m.astype("Int64")

# %%
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(Ns, 2**Ns, color=rdp.estilo.TINTA, linestyle="--", marker="o", markersize=4)
ax.plot(Ns, tabla_m["Sauer k=4"], color=DORADO, linestyle=":", linewidth=1.8)
ax.plot(Ns, intervalos, color=AZUL, marker="o")
ax.plot(Ns, rayos, color=GRIS, marker="o")
ax.plot(range(1, 7), perceptron, color=ROJO, marker="o", linewidth=2.4)
ax.set(yscale="log", xlim=(0.5, 12.6), xticks=Ns,
       title="Función de crecimiento contada: polinomial, no exponencial",
       xlabel="número de puntos N", ylabel="$m_{\\mathcal{H}}(N)$  (escala log)")
ax.text(10.2, 2**10, "$2^N$", va="center", fontsize=12)
ax.text(10.2, tabla_m["Sauer k=4"].iloc[-1], "cota de Sauer\n(k = 4)", va="center", fontsize=10.5, color=DORADO)
ax.text(10.2, intervalos[-1], "intervalos", va="center", fontsize=11)
ax.text(10.2, rayos[-1], "rayos", va="center", fontsize=11)
ax.annotate("perceptrón R²", xy=(5, perceptron[4]), xytext=(3.0, 90), fontsize=11, color=ROJO,
            arrowprops={"arrowstyle": "-", "color": ROJO})
plt.show()

# %% [markdown]
# Mientras $\mathcal{H}$ rompe los puntos, $m_{\mathcal{H}}(N)$ sigue a $2^N$; en cuanto aparece el punto de quiebre, la curva se despega y, en escala logarítmica, se vuelve cada vez más plana. Eso no es casualidad: es el siguiente resultado.

# %% [markdown]
# ## 7. El lema de Sauer
#
# > **Lema de Sauer.** Si $\mathcal{H}$ tiene un punto de quiebre $k$, entonces para todo $N$
# > $$
# > m_{\mathcal{H}}(N) \;\le\; \sum_{i=0}^{k-1} \binom{N}{i}.
# > $$
#
# El lado derecho es un **polinomio en $N$ de grado $k - 1 = d_{\text{VC}}$**; una cota más simple y muy usada es $m_{\mathcal{H}}(N) \le N^{d_{\text{VC}}} + 1$.
#
# **Idea de la demostración.** Sea $B(N, k)$ el máximo número de vectores en $\{-1, +1\}^N$ que se pueden tener sin que ningún subconjunto de $k$ coordenadas quede roto. Separando los vectores según su última coordenada, los que aparecen con ambos valores de la última coordenada no pueden romper $k - 1$ de las primeras $N - 1$ (si lo hicieran, junto con la última romperían $k$). Eso da la recursión $B(N, k) \le B(N-1, k) + B(N-1, k-1)$, la misma de los coeficientes binomiales, y por inducción $B(N, k) \le \sum_{i<k}\binom{N}{i}$.
#
# La consecuencia es enorme: **basta un solo punto de quiebre para que la función de crecimiento pase de exponencial a polinomial.** No hay término medio. Comparemos con lo contado:

# %%
comparacion = pd.DataFrame({
    "rayos": rayos, "Sauer k=2": tabla_m["Sauer k=2"].to_numpy(),
    "intervalos": intervalos, "Sauer k=3": tabla_m["Sauer k=3"].to_numpy(),
    "perceptrón R²": tabla_m["perceptrón R²"].to_numpy(), "Sauer k=4": tabla_m["Sauer k=4"].to_numpy(),
}, index=pd.Index(Ns, name="N")).astype("Int64")
comparacion

# %% [markdown]
# Para rayos e intervalos la cota de Sauer **se alcanza exactamente**: $1 + N$ y $1 + N + \binom{N}{2}$. Para el perceptrón es solo una cota: en $N = 6$ contamos 32 dicotomías y la cota permite 42. (El conteo exacto, debido a Cover en 1965, es $2\sum_{i=0}^{2}\binom{N-1}{i}$ para puntos en posición general; compruébalo con la tabla.)

# %% [markdown]
# ## 8. La cota VC frente a la brecha real
#
# El paso final, cuya demostración es bastante más larga y no haremos aquí, es justificar que en la cota de la unión se puede reemplazar $M$ por la función de crecimiento (con algunos cambios en las constantes). El resultado es la **desigualdad de Vapnik–Chervonenkis**:
#
# > **Cota de generalización VC.** Para cualquier tolerancia $\delta > 0$, con probabilidad al menos $1 - \delta$ sobre la muestra,
# > $$
# > E_{\text{out}}(g) \;\le\; E_{\text{in}}(g) + \underbrace{\sqrt{\frac{8}{N}\ln\frac{4\,m_{\mathcal{H}}(2N)}{\delta}}}_{\Omega(N,\, \mathcal{H},\, \delta)},
# > $$
# > y esto vale para **cualquier** $g \in \mathcal{H}$, sin importar el algoritmo que la eligió ni la distribución de los datos ni la función objetivo.
#
# Con $m_{\mathcal{H}}(2N) \le (2N)^{d_{\text{VC}}} + 1$, el logaritmo crece como $d_{\text{VC}} \ln N$ y entonces $\Omega \approx \sqrt{8\, d_{\text{VC}} \ln N / N} \to 0$. **Si $d_{\text{VC}}$ es finita, aprender es posible**: con suficientes datos, $E_{\text{in}}$ es un buen indicador de $E_{\text{out}}$.
#
# ¿Qué tan buena es la cota en la práctica? Volvamos al problema sintético de la sección 3 (puntos uniformes en el cuadrado, objetivo lineal, 10 % de ruido). Ahora sí **aprendemos**: ajustamos una regresión logística, cuyo clasificador final $\operatorname{sign}(\mathbf{w}\cdot\mathbf{x} + b)$ es un perceptrón en $\mathbb{R}^2$ ($d_{\text{VC}} = 3$), medimos su $E_{\text{in}}$ en los datos de entrenamiento y calculamos su $E_{\text{out}}$ exacto con las áreas. Repetimos con muchas muestras para cada $N$.

# %%
def cota_vc(N, d_vc=3, delta=0.05):
    return np.sqrt(8 / N * np.log(4 * ((2 * N) ** d_vc + 1) / delta))


Ns_vc = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
REPS = 200
brechas = {}
for N in Ns_vc:
    b = []
    for _ in range(REPS):
        X, y = muestra(N, rng)
        if len(np.unique(y)) < 2:
            continue
        modelo = LogisticRegression(C=1e4, max_iter=1000).fit(X, y)
        w, b0 = modelo.coef_[0], modelo.intercept_[0]
        b.append(e_out(w, b0) - e_in(w, b0, X, y))
    brechas[N] = np.array(b)

resumen = pd.DataFrame({
    "brecha promedio |E_in − E_out|": [np.mean(np.abs(brechas[N])) for N in Ns_vc],
    "percentil 95 de E_out − E_in": [np.quantile(brechas[N], 0.95) for N in Ns_vc],
    "cota VC (δ = 0.05)": cota_vc(Ns_vc),
}, index=pd.Index(Ns_vc, name="N"))
resumen["cota / percentil 95"] = resumen["cota VC (δ = 0.05)"] / resumen["percentil 95 de E_out − E_in"]
resumen.round(4)

# %% [markdown]
# La cota VC con $\delta = 0.05$ promete que $E_{\text{out}} - E_{\text{in}}$ no rebasa $\Omega$ en el 95 % de las muestras, así que la comparación justa es con el **percentil 95** de la brecha simulada, no con su promedio.

# %%
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(Ns_vc, resumen["cota VC (δ = 0.05)"], color=rdp.estilo.TINTA, linestyle="--", marker="o", markersize=4)
ax.plot(Ns_vc, resumen["percentil 95 de E_out − E_in"], color=ROJO, marker="o")
ax.plot(Ns_vc, resumen["brecha promedio |E_in − E_out|"], color=AZUL, marker="o")
ax.axhline(1, color=rdp.estilo.TENUE, linewidth=0.8)
ax.axvline(30, color=DORADO, linewidth=1.2, linestyle=":")
ax.text(33, 0.0035, "$N = 10\\,d_{VC}$", color=DORADO, fontsize=11)
ax.set(xscale="log", yscale="log", xlim=(7, 40000), ylim=(2e-3, 5),
       title="Cota VC contra la brecha real (perceptrón en R², $d_{VC}$ = 3)",
       xlabel="número de ejemplos N (escala log)", ylabel="brecha entre $E_{out}$ y $E_{in}$ (escala log)")
for nombre, col, color in (("cota VC", "cota VC (δ = 0.05)", rdp.estilo.TINTA),
                           ("percentil 95", "percentil 95 de E_out − E_in", ROJO),
                           ("promedio", "brecha promedio |E_in − E_out|", AZUL)):
    ax.text(11500, resumen[col].iloc[-1], nombre, va="center", fontsize=11, color=color)
plt.show()

# %% [markdown]
# Lo que muestra la gráfica:
#
# - **La cota se cumple siempre, pero es muy holgada**: es entre 10 y 30 veces mayor que la brecha real (columna «cota / percentil 95»). Para $N = 100$ la cota supera 1, así que no dice nada: un error siempre es menor que 1.
# - **La tendencia sí es la correcta.** En escala log-log, la cota y la brecha real bajan casi en paralelo, con pendiente cercana a $-1/2$: la brecha se reduce como $1/\sqrt{N}$ (salvo un factor logarítmico en la cota).
#
# ¿Por qué tan holgada? Cada paso de la demostración sacrifica algo: Hoeffding ya es holgada (sección 2); la cota de la unión, aunque ahora sobre dicotomías, sigue sumando eventos traslapados (sección 5); y $m_{\mathcal{H}}(2N)$ es el **peor caso** sobre todas las distribuciones y todos los conjuntos de puntos, mientras que nuestro problema es uno particular y bastante amable. La cota VC es universal, y la universalidad se paga.
#
# **¿Cuántos datos pide la teoría?** Si queremos $\Omega \le \varepsilon$ con probabilidad $1 - \delta$, hay que despejar $N$ de $N = \frac{8}{\varepsilon^2}\ln\frac{4((2N)^{d_{\text{VC}}}+1)}{\delta}$, lo que se hace por iteración:

# %%
def n_necesario(eps, d_vc, delta=0.05):
    N = 1000.0
    for _ in range(100):
        N = 8 / eps**2 * np.log(4 * ((2 * N) ** d_vc + 1) / delta)
    return int(np.ceil(N))


pd.DataFrame({f"d_VC = {d}": [n_necesario(0.1, d)] for d in (1, 2, 3, 5, 10)},
             index=["N para Ω ≤ 0.1 (δ = 0.05)"])

# %% [markdown]
# Para $d_{\text{VC}} = 3$ la teoría pide unos 30,000 ejemplos para garantizar una brecha de 0.1, y crece casi proporcionalmente con $d_{\text{VC}}$ (unos 10,000 ejemplos por cada unidad de $d_{\text{VC}}$). En la simulación, con $N = 100$ la brecha real ya está por debajo de 0.1 en el 95 % de los casos. Por eso la regla práctica que se usa es mucho más modesta: **$N \ge 10\, d_{\text{VC}}$** suele bastar para generalizar razonablemente. Lo valioso de la cota no es su valor numérico, sino lo que dice cualitativamente: el número de datos necesario crece con la capacidad del modelo, medida por $d_{\text{VC}}$, aproximadamente en proporción.

# %% [markdown]
# ## 9. Qué garantiza la teoría y qué no
#
# La teoría VC responde la pregunta del principio: **si un modelo acierta en los datos que vio, acertará en datos nuevos siempre que los datos sean muestras independientes de la misma población y el conjunto de hipótesis no sea demasiado rico para la cantidad de datos.** Pero conviene tener claro lo que **no** dice:
#
# - **No dice que $E_{\text{in}}$ vaya a ser pequeño.** Garantiza que $E_{\text{out}} \approx E_{\text{in}}$; que $E_{\text{in}}$ sea bajo depende de que $\mathcal{H}$ contenga alguna hipótesis parecida a $f$, y de que el algoritmo la encuentre. Una recta nunca aproximará bien una frontera circular, por muchos datos que tengamos.
# - **No dice nada si los datos no son independientes o si los datos nuevos vienen de otra distribución.** Fotos de un hospital para un modelo que se usará en otro, o datos de un año para predecir otro, rompen la hipótesis de la urna fija.
# - **Elegir mirando los datos cuenta.** Si probamos 20 modelos y nos quedamos con el de menor error en un conjunto de prueba, ese error ya no es una estimación honesta (es el $c_{\min}$ de la sección 4). Esto será central en la libreta de validación.
#
# De aquí sale el **compromiso entre aproximación y generalización**. Escrito como
# $$
# E_{\text{out}}(g) \le \underbrace{E_{\text{in}}(g)}_{\text{baja si } \mathcal{H} \text{ es rico}} + \underbrace{\Omega(N, \mathcal{H}, \delta)}_{\text{sube si } \mathcal{H} \text{ es rico}},
# $$
# un $\mathcal{H}$ más complejo aproxima mejor (baja $E_{\text{in}}$) pero generaliza peor (sube $\Omega$), y el mejor $E_{\text{out}}$ está en un punto intermedio que depende de $N$. Las siguientes libretas miran el mismo compromiso desde otro ángulo (sesgo y varianza) y con herramientas prácticas (validación).
#
# | Resultado | Qué afirma | Hipótesis | Lo que vimos en la simulación |
# |---|---|---|---|
# | Hoeffding | $\mathbb{P}[\lvert\nu-\mu\rvert>\varepsilon] \le 2e^{-2\varepsilon^2N}$ | muestras independientes, $\mu$ fija antes de muestrear | se cumple siempre; holgada pero con la forma correcta |
# | Hipótesis fija | $E_{\text{in}}(h) \approx E_{\text{out}}(h)$ | $h$ elegida sin ver los datos | permite verificar, no aprender |
# | Elegir la mejor | Hoeffding deja de valer para $g$ | — | $c_{\min}$ viola la cota por mucho |
# | Cota de la unión | $\mathbb{P}[\dots] \le 2Me^{-2\varepsilon^2N}$ | $\mathcal{H}$ finito | muy pesimista con hipótesis parecidas; inútil si $M = \infty$ |
# | Lema de Sauer | $m_{\mathcal{H}}(N) \le \sum_{i<k}\binom{N}{i}$ | punto de quiebre $k$ | exacta para rayos e intervalos; holgada para el perceptrón |
# | Cota VC | $E_{\text{out}} \le E_{\text{in}} + \sqrt{\tfrac{8}{N}\ln\tfrac{4m_{\mathcal{H}}(2N)}{\delta}}$ | muestras independientes, $d_{\text{VC}}$ finita | se cumple, 10 a 30 veces holgada, misma tendencia $1/\sqrt{N}$ |

# %% [markdown]
# ## 10. Autoevaluación

# %%
rdp.quiz.mostrar("u03", ids=["u03-p01", "u03-p02", "u03-p03", "u03-p04", "u03-p05"])

# %% [markdown]
# ## 11. Ejercicios
#
# 1. En la sección 2, cambia $\mu$ a 0.05. ¿La probabilidad real de $|\nu - \mu| > 0.05$ con $N = 100$ se acerca o se aleja de la cota de Hoeffding? ¿Por qué la cota no cambia? (Pista: la varianza de $\nu$ es $\mu(1-\mu)/N$.)
# 2. Repite el experimento de las monedas con 10, 100 y 10,000 monedas. ¿Cómo cambia la distribución de $\nu_{\min}$? Calcula la cota de la unión $2Me^{-2\varepsilon^2 N}$ para $\varepsilon = 0.4$: ¿cuál es el mayor $M$ para el que todavía dice algo?
# 3. Usa `separable` para contar las dicotomías de 4 puntos en los que uno está dentro del triángulo de los otros tres. ¿Cuáles son las dos imposibles? Haz lo mismo con 4 puntos donde tres están alineados: ¿obtienes 14 o menos? ¿Contradice esto que $m_{\mathcal{H}}(4) = 14$?
# 4. Cuenta las dicotomías de los perceptrones en $\mathbb{R}^3$ (con $b$) para $N = 1, \dots, 6$ puntos al azar. ¿Dónde está el punto de quiebre? ¿Coincide con $d_{\text{VC}} = d + 1$?
# 5. Considera los «círculos positivos» en $\mathbb{R}^2$: $h(\mathbf{x}) = +1$ si $\lVert \mathbf{x} - \mathbf{c} \rVert < r$. Argumenta que rompen 3 puntos. ¿Cuál crees que es su dimensión VC? (Pista: con la transformación $(x_1, x_2) \mapsto (x_1, x_2, x_1^2 + x_2^2)$ un círculo se vuelve un plano.)
# 6. En la sección 8, sustituye la regresión logística por un modelo con frontera cuadrática (agrega $x_1^2$, $x_1x_2$ y $x_2^2$ como variables, de modo que $d_{\text{VC}} = 6$). ¿Cómo cambian $E_{\text{in}}$, $E_{\text{out}}$ y la brecha para $N = 20$ y para $N = 1000$? ¿Qué dice la cota VC?
