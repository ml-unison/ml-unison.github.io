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
# # Sesgo, varianza y regularización
#
# **Unidad 3 · Ejemplo 2**
#
# En la Unidad 1 ajustamos polinomios de grado 1, 4 y 12 a quince puntos con ruido y vimos dos maneras de fracasar: el grado 1 no alcanza a seguir la curva y el grado 12 la sigue demasiado, ruido incluido. Dijimos que había que buscar un punto medio. En esta libreta vamos a darle forma matemática a esa intuición y a usarla:
#
# 1. **derivar** que el error esperado fuera de la muestra se descompone exactamente en tres términos: sesgo², varianza y ruido;
# 2. **medir** esos tres términos en una simulación donde conocemos la función verdadera, y ver de dónde sale la famosa curva en forma de U;
# 3. leer las **curvas de aprendizaje**, que dicen qué pasa cuando conseguimos más datos;
# 4. estudiar la **regularización** (*ridge* y *lasso*) como una manera de controlar la varianza sin cambiar de conjunto de hipótesis;
# 5. aplicarlo a datos reales de pacientes con diabetes.
#
# A diferencia de la cota VC, que vale para cualquier distribución y por eso es pesimista, el análisis de sesgo y varianza es un **promedio**: dice qué esperar «en un día típico» si repitiéramos el experimento muchas veces. Lo pagamos con una restricción: el análisis es para el error cuadrático, es decir, para regresión.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado.

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, lasso_path
from sklearn.metrics import mean_squared_error, r2_score

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS

# %% [markdown]
# ## 1. La descomposición sesgo-varianza
#
# ### La pregunta
#
# Fijemos un conjunto de hipótesis $\mathcal{H}$ (por ejemplo, los polinomios de grado 4) y un algoritmo (mínimos cuadrados). El resultado del aprendizaje depende del conjunto de entrenamiento $\mathcal{D}$ que nos tocó: con otros 15 puntos habríamos obtenido otro polinomio. Por eso escribimos $g^{(\mathcal{D})}$ para la hipótesis final. Su error fuera de la muestra, con error cuadrático, es
#
# $$
# E_{\text{out}}\big(g^{(\mathcal{D})}\big) = \mathbb{E}_{\mathbf{x},y}\Big[\big(g^{(\mathcal{D})}(\mathbf{x}) - y\big)^2\Big].
# $$
#
# Es una variable aleatoria: depende de $\mathcal{D}$. La pregunta es cuánto vale **en promedio** sobre todos los conjuntos de entrenamiento de tamaño $N$ que pudieron tocarnos: $\mathbb{E}_{\mathcal{D}}\big[E_{\text{out}}(g^{(\mathcal{D})})\big]$.
#
# ### El modelo de los datos
#
# Suponemos que las etiquetas son una función desconocida más ruido:
#
# $$
# y = f(\mathbf{x}) + \varepsilon, \qquad \mathbb{E}[\varepsilon \mid \mathbf{x}] = 0, \qquad \operatorname{Var}(\varepsilon \mid \mathbf{x}) = \sigma^2,
# $$
#
# y que el punto de prueba $(\mathbf{x}, y)$ es independiente de $\mathcal{D}$. Definimos la **hipótesis promedio**
#
# $$
# \bar g(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}\big[g^{(\mathcal{D})}(\mathbf{x})\big],
# $$
#
# que es lo que obtendríamos si pudiéramos entrenar con infinitos conjuntos de datos y promediar las respuestas. Ojo: $\bar g$ no necesariamente pertenece a $\mathcal{H}$, y no es algo que podamos calcular en la práctica; es un instrumento teórico.
#
# ### La derivación
#
# **Paso 1: separar el ruido.** Para $\mathbf{x}$ fija, escribimos $g^{(\mathcal{D})}(\mathbf{x}) - y = \big(g^{(\mathcal{D})}(\mathbf{x}) - f(\mathbf{x})\big) - \varepsilon$ y desarrollamos el cuadrado:
#
# $$
# \mathbb{E}_{\mathcal{D},\varepsilon}\Big[\big(g^{(\mathcal{D})}(\mathbf{x}) - y\big)^2\Big]
# = \mathbb{E}_{\mathcal{D}}\Big[\big(g^{(\mathcal{D})}(\mathbf{x}) - f(\mathbf{x})\big)^2\Big]
# - 2\,\mathbb{E}_{\mathcal{D}}\big[g^{(\mathcal{D})}(\mathbf{x}) - f(\mathbf{x})\big]\,\underbrace{\mathbb{E}[\varepsilon \mid \mathbf{x}]}_{=\,0}
# + \underbrace{\mathbb{E}[\varepsilon^2 \mid \mathbf{x}]}_{=\,\sigma^2}.
# $$
#
# El producto cruzado se factoriza porque el ruido del punto de prueba es independiente de $\mathcal{D}$, y se anula porque el ruido tiene media cero.
#
# **Paso 2: sumar y restar $\bar g$.** En el primer término:
#
# $$
# \begin{aligned}
# \mathbb{E}_{\mathcal{D}}\Big[\big(g^{(\mathcal{D})}(\mathbf{x}) - f(\mathbf{x})\big)^2\Big]
# &= \mathbb{E}_{\mathcal{D}}\Big[\big(g^{(\mathcal{D})}(\mathbf{x}) - \bar g(\mathbf{x}) + \bar g(\mathbf{x}) - f(\mathbf{x})\big)^2\Big] \\
# &= \mathbb{E}_{\mathcal{D}}\Big[\big(g^{(\mathcal{D})}(\mathbf{x}) - \bar g(\mathbf{x})\big)^2\Big]
# + 2\big(\bar g(\mathbf{x}) - f(\mathbf{x})\big)\underbrace{\mathbb{E}_{\mathcal{D}}\big[g^{(\mathcal{D})}(\mathbf{x}) - \bar g(\mathbf{x})\big]}_{=\,\bar g(\mathbf{x}) - \bar g(\mathbf{x})\,=\,0}
# + \big(\bar g(\mathbf{x}) - f(\mathbf{x})\big)^2.
# \end{aligned}
# $$
#
# Aquí el término cruzado se anula **por la definición** de $\bar g$: $\bar g(\mathbf{x}) - f(\mathbf{x})$ no depende de $\mathcal{D}$ y sale de la esperanza.
#
# **Paso 3: promediar sobre $\mathbf{x}$.** Juntando todo e intercambiando el orden de las esperanzas (todo es no negativo):
#
# $$
# \boxed{\;
# \mathbb{E}_{\mathcal{D}}\big[E_{\text{out}}(g^{(\mathcal{D})})\big]
# = \underbrace{\mathbb{E}_{\mathbf{x}}\Big[\big(\bar g(\mathbf{x}) - f(\mathbf{x})\big)^2\Big]}_{\text{sesgo}^2}
# + \underbrace{\mathbb{E}_{\mathbf{x}}\Big[\mathbb{E}_{\mathcal{D}}\big[\big(g^{(\mathcal{D})}(\mathbf{x}) - \bar g(\mathbf{x})\big)^2\big]\Big]}_{\text{varianza}}
# + \underbrace{\sigma^2}_{\text{ruido}}
# \;}
# $$
#
# ### Qué significa cada término
#
# - **Sesgo²**: qué tan lejos está de $f$ la respuesta *promedio*. Mide la capacidad de $\mathcal{H}$ (y del algoritmo) para aproximar $f$; no mejora con más datos de entrenamiento, porque $\bar g$ apenas cambia con $N$.
# - **Varianza**: cuánto se mueve la respuesta de un conjunto de entrenamiento a otro. Mide la inestabilidad; baja cuando $N$ crece.
# - **Ruido**: el error que cometería incluso la $f$ verdadera. Ningún algoritmo puede bajarlo; es el piso.
#
# Un $\mathcal{H}$ más grande suele bajar el sesgo (hay hipótesis más cercanas a $f$) y subir la varianza (hay más maneras de ajustarse al ruido particular de $\mathcal{D}$). Es la misma tensión que en la cota VC entre $E_{\text{in}}$ y el término de complejidad, vista desde otro ángulo.

# %% [markdown]
# ## 2. La simulación: polinomios y $\sin(2\pi x)$
#
# ### El problema y los datos
#
# Para medir el sesgo necesitamos conocer $f$, y eso solo pasa en una simulación. Retomamos el experimento de la Unidad 1:
#
# - $x$ en $[0, 1]$, repartida al azar pero sin dejar huecos grandes (abajo explicamos por qué),
# - $f(x) = \sin(2\pi x)$,
# - ruido gaussiano con $\sigma = 0.25$, así que el piso es $\sigma^2 = 0.0625$,
# - conjuntos de entrenamiento de $N = 15$ puntos.
#
# Generamos $R = 2000$ conjuntos de entrenamiento distintos y, para cada grado $Q = 0, 1, \dots, 12$, ajustamos un polinomio por mínimos cuadrados a cada uno. Así obtenemos 2000 hipótesis $g^{(\mathcal{D})}$ por grado; su promedio aproxima $\bar g$ y su dispersión, la varianza. Las esperanzas sobre $x$ se aproximan con 2000 puntos de prueba independientes.
#
# **Cómo sorteamos las $x$.** Si las 15 $x$ se sortean uniformes e independientes, de vez en cuando quedan todas lejos de un extremo del intervalo, y un polinomio de grado alto, obligado a **extrapolar**, se dispara a valores de miles o millones. Esos conjuntos raros dominan el promedio (lo comprobamos abajo). Para que la simulación muestre el fenómeno típico y no esos accidentes, usamos un muestreo **estratificado**: partimos $[0,1]$ en 15 intervalos iguales y sorteamos una $x$ uniforme dentro de cada uno. Los puntos siguen siendo aleatorios, pero cubren todo el intervalo.
#
# **Un detalle numérico.** Los polinomios de grado alto escritos como $w_0 + w_1 x + \dots + w_Q x^Q$ con $x \in [0, 1]$ dan sistemas muy mal condicionados ($x^{11}$ y $x^{12}$ son casi la misma columna). Usamos la variable $z = 2x - 1 \in [-1, 1]$: los polinomios de grado $Q$ en $z$ son exactamente los mismos que en $x$, así que $\mathcal{H}$ no cambia, pero los cálculos son mucho más estables.

# %%
SIGMA, N, R = 0.25, 15, 2000
GRADOS = np.arange(13)


def f(x):
    return np.sin(2 * np.pi * x)


def caracteristicas(x, grado):
    """Matriz de diseño [1, z, z², …, z^grado] con z = 2x − 1; funciona con lotes de conjuntos."""
    return (2 * x[..., None] - 1) ** np.arange(grado + 1)


def ajustar(x, y, grado, lam=0.0):
    """Mínimos cuadrados (o ridge, si lam > 0) para un lote de R conjuntos: x, y tienen forma (R, N).

    Ridge se resuelve como mínimos cuadrados aumentados: agregar las filas √λ·I con
    etiqueta 0 equivale a sumar λ‖w‖² al error. No se penaliza el término constante.
    Regresa los coeficientes, de forma (R, grado + 1).
    """
    Phi = caracteristicas(x, grado)
    if lam > 0:
        P = np.sqrt(lam) * np.eye(grado + 1)[1:]
        Phi = np.concatenate([Phi, np.broadcast_to(P, (len(x), grado, grado + 1))], axis=1)
        y = np.concatenate([y, np.zeros((len(x), grado))], axis=1)
    return (np.linalg.pinv(Phi) @ y[..., None])[..., 0]


rng = np.random.default_rng(SEMILLA)


def sortear_x(R, n, rng):
    """R conjuntos de n puntos en [0, 1], uno uniforme en cada intervalo [i/n, (i+1)/n)."""
    return (np.arange(n) + rng.uniform(0, 1, (R, n))) / n


X_ent = sortear_x(R, N, rng)
Y_ent = f(X_ent) + SIGMA * rng.normal(size=(R, N))
x_prueba = rng.uniform(0, 1, 2000)
y_prueba = f(x_prueba) + SIGMA * rng.normal(size=x_prueba.size)


def descomponer(W, grado):
    """Sesgo², varianza y E_out medido a partir de los coeficientes de las R hipótesis."""
    G = caracteristicas(x_prueba, grado) @ W.T  # (puntos de prueba, R): g^D(x) para cada D
    g_barra = G.mean(axis=1)
    return {
        "sesgo2": np.mean((g_barra - f(x_prueba)) ** 2),
        "varianza": np.mean(G.var(axis=1)),
        "E_out": np.mean((G - y_prueba[:, None]) ** 2),
        "E_in": None,
    }


filas = []
for Q in GRADOS:
    W = ajustar(X_ent, Y_ent, Q)
    d = descomponer(W, Q)
    d["E_in"] = np.mean((np.einsum("rnk,rk->rn", caracteristicas(X_ent, Q), W) - Y_ent) ** 2)
    filas.append({"grado": Q, **d})
tabla = pd.DataFrame(filas).set_index("grado")
tabla["sesgo2 + var + σ²"] = tabla["sesgo2"] + tabla["varianza"] + SIGMA**2
tabla.round(4)

# %% [markdown]
# Las dos últimas columnas comparan el $E_{\text{out}}$ **medido** directamente con puntos de prueba ruidosos contra la **suma** de los tres términos. Coinciden hasta el tercer decimal (la pequeña diferencia es el ruido de la muestra de prueba): la descomposición no es una aproximación, es una identidad.
#
# Otros detalles de la tabla:
#
# - $E_{\text{in}}$ baja siempre con el grado, como debe ser, porque cada $\mathcal{H}$ contiene a la anterior. Para grado 12 es una fracción pequeña de $\sigma^2$: el polinomio se está comiendo el ruido.
# - El sesgo de grado 2 es igual al de grado 1. No es casualidad: $\sin(2\pi x)$ es antisimétrica alrededor de $x = 1/2$ (impar en $z$), así que el término $z^2$ no le ayuda; lo mismo pasa entre grados 3 y 4. Lo que sí sube es la varianza: un parámetro más que no sirve para nada solo agrega inestabilidad.

# %%
Q_opt = int(tabla["E_out"].idxmin())
print(f"grado con menor E_out esperado: {Q_opt}")
print(f"  sesgo² = {tabla.loc[Q_opt, 'sesgo2']:.4f}   varianza = {tabla.loc[Q_opt, 'varianza']:.4f}   "
      f"ruido = {SIGMA**2:.4f}   E_out = {tabla.loc[Q_opt, 'E_out']:.4f}")

# %% [markdown]
# **¿Y si las $x$ fueran completamente al azar?** Comparemos, para grado 9, el error de cada una de las 2000 hipótesis con los dos muestreos. Usamos el $E_{\text{out}}$ de cada $g^{(\mathcal{D})}$ y resumimos con la mediana y la media.

# %%
X_libre = rng.uniform(0, 1, (R, N))
Y_libre = f(X_libre) + SIGMA * rng.normal(size=(R, N))
for nombre, Xs, Ys in [("estratificado", X_ent, Y_ent), ("uniforme libre", X_libre, Y_libre)]:
    e_out_D = np.mean((caracteristicas(x_prueba, 9) @ ajustar(Xs, Ys, 9).T - y_prueba[:, None]) ** 2, axis=0)
    print(f"{nombre:>15}:  mediana = {np.median(e_out_D):8.3f}   media = {e_out_D.mean():12.4g}   máximo = {e_out_D.max():12.4g}")

# %% [markdown]
# Con $x$ libres, la hipótesis *típica* (la mediana) ya es bastante peor, y la media se va a valores absurdos por culpa de unos pocos conjuntos en los que el polinomio extrapola: el promedio lo dominan los accidentes. Es real, y es otra razón para desconfiar de los polinomios de grado alto, pero haría ilegible la gráfica; por eso seguimos con el muestreo estratificado.

# %%
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(GRADOS, tabla["sesgo2"], "o-", color=AZUL)
ax.plot(GRADOS, tabla["varianza"], "o-", color=ROJO)
ax.plot(GRADOS, tabla["E_out"], "o-", color=rdp.estilo.TINTA, lw=2.8)
ax.axhline(SIGMA**2, color=GRIS, ls="--", lw=1.4)
ax.axvline(Q_opt, color=DORADO, lw=1.2, ls=":")
ax.set_yscale("log")
ax.set(xlim=(-0.5, 13.5), ylim=(2e-5, 300), xticks=GRADOS,
       xlabel="grado del polinomio $Q$", ylabel="error cuadrático (escala log)",
       title=f"Sesgo y varianza con N = {N}: la curva en forma de U")
ax.text(12.3, tabla["sesgo2"].iloc[-1], "sesgo²", color=AZUL, va="center")
ax.text(5.2, 0.014, "varianza", color=ROJO, va="center")
ax.text(4.6, 0.32, "$E_{\\mathrm{out}}$ esperado", color=rdp.estilo.TINTA, va="center")
ax.text(-0.3, 0.042, "ruido $\\sigma^2$ (piso)", color=GRIS, va="center")
ax.text(Q_opt + 0.15, 80, f"mínimo: grado {Q_opt}", color=DORADO)
plt.show()

# %% [markdown]
# La gráfica cuenta la historia completa (el eje vertical es logarítmico para que quepan los grados altos):
#
# - Con grado 0 y 1 domina el **sesgo**: una recta no puede dibujar una onda, por más datos que tenga. El sesgo cae bruscamente en grado 3, cuando el polinomio ya puede tener los dos «dobleces» del seno.
# - A partir de ahí el sesgo es prácticamente cero (en grado 5 ya es del orden de $10^{-4}$) y lo único que cambia es la **varianza**, que crece sin control: con 15 puntos, un polinomio de grado 12 tiene 13 coeficientes y casi interpola los datos.
# - El $E_{\text{out}}$ esperado es la suma, con mínimo en un grado intermedio. Nunca baja del piso $\sigma^2$.
#
# **Un detalle de la estimación.** El sesgo² estimado vuelve a subir en grados 11 y 12, cuando en teoría debería seguir en cero. Es un artefacto de la simulación: estimamos $\bar g$ con el promedio de $R$ hipótesis, y ese promedio tiene su propia varianza, $\text{varianza}/R$, que se suma al sesgo² estimado. Para grado 12 eso es $59.7/2000 \approx 0.03$, justo el orden de lo que vemos. Con $R$ más grande ese falso sesgo desaparecería.
#
# Veamos qué significan «sesgo» y «varianza» en términos de las hipótesis mismas.

# %%
xs = np.linspace(0, 1, 400)
fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, Q, color in zip(axs, [1, 9], [AZUL, ROJO]):
    W = ajustar(X_ent, Y_ent, Q)
    G = caracteristicas(xs, Q) @ W.T
    ax.plot(xs, G[:, :60], color=color, alpha=0.18, lw=1)
    ax.plot(xs, G.mean(axis=1), color=color, lw=3)
    ax.plot(xs, f(xs), color=rdp.estilo.TINTA, ls="--", lw=2)
    d = tabla.loc[Q]
    ax.set(ylim=(-2.2, 2.2), xlabel="$x$",
           title=f"Grado {Q}: sesgo² = {d['sesgo2']:.3f}, varianza = {d['varianza']:.3f}")
axs[0].set_ylabel("$y$")
axs[0].text(0.62, 1.15, "60 hipótesis $g^{(\\mathcal{D})}$", color=AZUL, fontsize=11)
axs[0].text(0.03, -0.75, "$\\bar g$ (promedio)", color=AZUL, fontsize=12)
axs[0].text(0.10, 1.25, "$f(x) = \\sin 2\\pi x$", color=rdp.estilo.TINTA, fontsize=12)
axs[1].text(0.45, 1.6, "60 hipótesis $g^{(\\mathcal{D})}$", color=ROJO, fontsize=11)
axs[1].text(0.55, -1.25, "$\\bar g$ casi igual a $f$", color=ROJO, fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# - **Grado 1.** Las 60 rectas se parecen mucho entre sí (varianza baja), pero su promedio $\bar g$ es una recta descendente que no se parece nada al seno: el error está en el promedio, es **sesgo**.
# - **Grado 9.** El promedio $\bar g$ es prácticamente el seno (sesgo casi nulo), pero cada polinomio individual se dispara, sobre todo en los extremos, donde hay pocos puntos que lo sujeten: el error está en la dispersión, es **varianza**.
#
# Nota lo que esto implica: si pudiéramos promediar muchos polinomios de grado 9 entrenados con conjuntos distintos, tendríamos un modelo excelente. No podemos (tenemos un solo $\mathcal{D}$), pero la idea de reducir varianza promediando reaparece en los métodos de ensamble (*bagging*, bosques aleatorios) de la Unidad 5.

# %% [markdown]
# ## 3. Curvas de aprendizaje
#
# ### La pregunta
#
# Con 15 puntos el grado 9 pierde. ¿Y si tuviéramos 100? La descomposición sugiere que la varianza baja con $N$ y el sesgo no, así que el modelo complejo debería alcanzar y rebasar al simple. Las **curvas de aprendizaje** grafican los errores esperados contra $N$, con $\mathcal{H}$ fijo:
#
# $$
# N \;\mapsto\; \mathbb{E}_{\mathcal{D}}\big[E_{\text{in}}(g^{(\mathcal{D})})\big], \qquad N \;\mapsto\; \mathbb{E}_{\mathcal{D}}\big[E_{\text{out}}(g^{(\mathcal{D})})\big].
# $$
#
# ### Lo que se puede demostrar
#
# Para regresión lineal con $d + 1$ parámetros y ruido de varianza $\sigma^2$, cuando el modelo contiene a $f$ (y las $\mathbf{x}$ son fijas, en el caso de $E_{\text{in}}$), se cumple de manera exacta
#
# $$
# \mathbb{E}_{\mathcal{D}}[E_{\text{in}}] = \sigma^2\Big(1 - \frac{d+1}{N}\Big),
# \qquad
# \mathbb{E}_{\mathcal{D}}[E_{\text{out}}] \approx \sigma^2\Big(1 + \frac{d+1}{N}\Big).
# $$
#
# La primera sale de que el vector de residuos es la proyección del ruido sobre el complemento ortogonal del espacio columna de $\mathbf{X}$, que tiene dimensión $N - (d+1)$ (es el ejercicio 3.4 de *Learning From Data*). La moraleja: la brecha de generalización esperada es del orden de $2\sigma^2 (d+1)/N$, proporcional al número de parámetros e inversamente proporcional a $N$. Nada de esto requiere la cota VC.

# %%
TAMANOS = np.array([10, 12, 14, 17, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200])
R_CURVAS = 1000
rng = np.random.default_rng(SEMILLA + 1)
curvas = {}
for Q in [1, 9]:
    filas = []
    for n in TAMANOS:
        Xn = sortear_x(R_CURVAS, n, rng)
        Yn = f(Xn) + SIGMA * rng.normal(size=(R_CURVAS, n))
        W = ajustar(Xn, Yn, Q)
        e_in = np.mean((np.einsum("rnk,rk->rn", caracteristicas(Xn, Q), W) - Yn) ** 2)
        e_out = np.mean((caracteristicas(x_prueba, Q) @ W.T - y_prueba[:, None]) ** 2)
        filas.append((n, e_in, e_out))
    curvas[Q] = pd.DataFrame(filas, columns=["N", "E_in", "E_out"]).set_index("N")

fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, Q in zip(axs, [1, 9]):
    c = curvas[Q]
    ax.plot(c.index, c["E_out"], "o-", color=ROJO, ms=4)
    ax.plot(c.index, c["E_in"], "o-", color=AZUL, ms=4)
    ax.axhline(SIGMA**2, color=GRIS, ls="--", lw=1.4)
    ax.set(ylim=(0, 0.6), xlabel="tamaño del conjunto de entrenamiento $N$",
           title=f"Grado {Q} ({Q + 1} parámetros)")
    ax.text(202, c["E_out"].iloc[-1] + 0.02, "$E_{\\mathrm{out}}$", color=ROJO, va="bottom")
    ax.text(202, c["E_in"].iloc[-1] - 0.02, "$E_{\\mathrm{in}}$", color=AZUL, va="top")
    ax.set_xlim(0, 225)
axs[0].set_ylabel("error cuadrático esperado")
axs[0].text(110, SIGMA**2 - 0.03, "ruido $\\sigma^2$", color=GRIS)
plt.tight_layout()
plt.show()

print("E_out esperado (grado 1 vs grado 9):")
print(pd.DataFrame({"grado 1": curvas[1]["E_out"], "grado 9": curvas[9]["E_out"]}).loc[[10, 14, 20, 30, 50, 100, 200]].round(3))

# %% [markdown]
# Las dos curvas de cada panel se acercan a un mismo valor, $\sigma^2 + \text{sesgo}^2$, pero cuentan historias distintas:
#
# - **Grado 1** converge muy pronto (con $N \approx 30$ ya casi no cambia), pero a un nivel alto: alrededor de $0.26$, de los cuales $0.195$ son sesgo y $0.0625$ ruido. **Conseguir más datos no sirve**; lo que hace falta es un modelo más rico.
# - **Grado 9** empieza fatal (con $N = 10$ interpola y $E_{\text{in}} = 0$) pero su brecha se cierra y converge al piso $\sigma^2$, porque su sesgo es casi nulo. Desde $N \approx 17$ ya es mejor que la recta.
#
# **Cómo se usa esto en la práctica.** Con datos reales no conocemos $\sigma^2$, pero sí podemos trazar las curvas con validación (lo haremos en la libreta 3): si $E_{\text{in}}$ y $E_{\text{val}}$ están juntas y altas, el problema es sesgo (cambia de modelo); si hay una brecha grande, el problema es varianza (consigue datos o regulariza).

# %% [markdown]
# ## 4. Regularización ridge
#
# ### La idea
#
# Muchas veces no podemos conseguir más datos. La otra manera de bajar la varianza es **restringir** el aprendizaje: seguimos usando polinomios de grado 12, pero les prohibimos tener coeficientes enormes. Los polinomios del panel derecho de arriba se disparan precisamente porque sus coeficientes son grandes y se cancelan entre sí.
#
# ### Versión restringida y versión penalizada
#
# Escribamos el error dentro de la muestra como $\lVert \mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2$, con $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$. La **versión restringida** de ridge es
#
# $$
# \min_{\mathbf{w}} \;\lVert \mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 \quad \text{sujeto a} \quad \lVert \mathbf{w}\rVert^2 \le C,
# $$
#
# que es literalmente un conjunto de hipótesis más pequeño $\mathcal{H}(C) \subset \mathcal{H}$. La **versión penalizada** es
#
# $$
# \min_{\mathbf{w}} \;\lVert \mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 + \lambda \lVert \mathbf{w}\rVert^2, \qquad \lambda \ge 0.
# $$
#
# **Son equivalentes.** Sea $\mathbf{w}_{\text{lin}}$ la solución de mínimos cuadrados. Si $\lVert\mathbf{w}_{\text{lin}}\rVert^2 \le C$, la restricción no estorba y corresponde a $\lambda = 0$. Si no, la solución restringida $\mathbf{w}^\ast$ está en la frontera $\lVert\mathbf{w}\rVert^2 = C$ (el error es convexo, y si el óptimo estuviera en el interior sería un mínimo sin restricción). En la frontera, el gradiente del error debe ser perpendicular a la esfera; si tuviera una componente tangente, podríamos movernos sobre la esfera y bajar el error. La normal a la esfera en $\mathbf{w}^\ast$ es $\mathbf{w}^\ast$ y el error debe crecer al salir de ella, así que
#
# $$
# \nabla_{\mathbf{w}} \lVert \mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 \Big|_{\mathbf{w}^\ast} = -2\lambda\,\mathbf{w}^\ast \quad\text{para algún } \lambda > 0,
# $$
#
# que es exactamente la condición de óptimo del problema penalizado (ver abajo). A cada $C$ le corresponde un $\lambda$ y viceversa; $C$ grande equivale a $\lambda$ pequeño. (Es el multiplicador de Lagrange de la restricción.)
#
# ### La solución cerrada
#
# La función $E(\mathbf{w}) = \lVert \mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 + \lambda \mathbf{w}^\top\mathbf{w}$ es cuadrática. Su gradiente es
#
# $$
# \nabla E(\mathbf{w}) = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) + 2\lambda\mathbf{w} = 2\big[(\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I})\mathbf{w} - \mathbf{X}^\top\mathbf{y}\big],
# $$
#
# y su hessiana, $2(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})$, es definida positiva para $\lambda > 0$: para todo $\mathbf{v} \ne 0$, $\mathbf{v}^\top(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\mathbf{v} = \lVert\mathbf{X}\mathbf{v}\rVert^2 + \lambda\lVert\mathbf{v}\rVert^2 > 0$. Así que $E$ es estrictamente convexa, la matriz es invertible y el único mínimo es
#
# $$
# \boxed{\;\mathbf{w}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}\;}
# $$
#
# Un bono: esto funciona **aunque $\mathbf{X}^\top\mathbf{X}$ sea singular** (más parámetros que datos, o columnas colineales), donde mínimos cuadrados no tiene solución única.
#
# ### Por qué baja la varianza (y sube el sesgo)
#
# Con la descomposición en valores singulares $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$ ($d_1 \ge d_2 \ge \dots \ge 0$) las predicciones en los datos quedan
#
# $$
# \mathbf{X}\mathbf{w}_{\text{ridge}} = \sum_j \mathbf{u}_j \,\frac{d_j^2}{d_j^2 + \lambda}\, \mathbf{u}_j^\top\mathbf{y}.
# $$
#
# Mínimos cuadrados ($\lambda = 0$) proyecta $\mathbf{y}$ sobre cada dirección $\mathbf{u}_j$ con factor 1. Ridge **encoge** cada componente por el factor $d_j^2/(d_j^2+\lambda) < 1$, y encoge mucho más las direcciones con $d_j$ pequeño: aquellas en las que los datos casi no varían, que son justo las que el ruido mueve más. Encoger reduce la varianza; pero como también encoge la señal, $\mathbb{E}[\mathbf{w}_{\text{ridge}}] = (\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}\,\mathbf{w}^\ast \ne \mathbf{w}^\ast$: el estimador queda **sesgado**. Regularizar es cambiar sesgo por varianza a propósito.
#
# **En la práctica** no se penaliza el término constante $w_0$ (no queremos que el modelo prefiera predecir cerca de cero), y las variables se estandarizan antes, porque la penalización $\lVert\mathbf{w}\rVert^2$ trata igual a todos los coeficientes y eso solo tiene sentido si están en la misma escala.
#
# ### La simulación con grado 12
#
# Volvemos al experimento: polinomios de grado 12, $N = 15$, y ahora $\lambda$ variable.

# %%
Q_RIDGE = 12
LAMBDAS = np.logspace(-8, 2, 31)
filas = []
for lam in LAMBDAS:
    W = ajustar(X_ent, Y_ent, Q_RIDGE, lam)
    filas.append({"lambda": lam, **descomponer(W, Q_RIDGE)})
tabla_ridge = pd.DataFrame(filas).drop(columns="E_in").set_index("lambda")
lam_opt = tabla_ridge["E_out"].idxmin()

print(f"Grado {Q_RIDGE} sin regularizar: sesgo² = {tabla.loc[Q_RIDGE, 'sesgo2']:.4f}, "
      f"varianza = {tabla.loc[Q_RIDGE, 'varianza']:.4f}, E_out = {tabla.loc[Q_RIDGE, 'E_out']:.4f}")
print(f"Grado {Q_RIDGE} con λ = {lam_opt:.1e}: sesgo² = {tabla_ridge.loc[lam_opt, 'sesgo2']:.4f}, "
      f"varianza = {tabla_ridge.loc[lam_opt, 'varianza']:.4f}, E_out = {tabla_ridge.loc[lam_opt, 'E_out']:.4f}")
print(f"Mejor grado sin regularizar ({Q_opt}): E_out = {tabla.loc[Q_opt, 'E_out']:.4f}")

# %%
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(LAMBDAS, tabla_ridge["sesgo2"], "o-", color=AZUL, ms=4)
ax.plot(LAMBDAS, tabla_ridge["varianza"], "o-", color=ROJO, ms=4)
ax.plot(LAMBDAS, tabla_ridge["E_out"], "o-", color=rdp.estilo.TINTA, ms=4, lw=2.8)
ax.axhline(SIGMA**2, color=GRIS, ls="--", lw=1.4)
ax.axvline(lam_opt, color=DORADO, ls=":", lw=1.2)
ax.set(xscale="log", yscale="log", xlim=(LAMBDAS[0] / 2, LAMBDAS[-1] * 60), ylim=(1e-4, 30),
       xlabel="$\\lambda$ (escala log)", ylabel="error cuadrático (escala log)",
       title=f"Ridge con polinomios de grado {Q_RIDGE} y N = {N}")
ax.text(LAMBDAS[-1] * 1.4, tabla_ridge["sesgo2"].iloc[-1], "sesgo²", color=AZUL, va="center")
ax.text(LAMBDAS[-1] * 1.4, tabla_ridge["varianza"].iloc[-1], "varianza", color=ROJO, va="center")
ax.text(LAMBDAS[-1] * 1.4, tabla_ridge["E_out"].iloc[-1] * 2.6, "$E_{\\mathrm{out}}$", color=rdp.estilo.TINTA, va="center")
ax.text(LAMBDAS[0], SIGMA**2 * 1.25, "ruido $\\sigma^2$", color=GRIS)
ax.text(lam_opt * 1.4, 12, f"$\\lambda^\\ast \\approx$ {lam_opt:.0e}", color=DORADO)
plt.show()

# %% [markdown]
# (El sesgo² que se ve con $\lambda$ diminuto es otra vez el artefacto $\text{varianza}/R$ de la sección 2.)
#
# Es la misma U de la sección 2, pero recorrida en sentido contrario: ahora el eje horizontal **baja** la complejidad efectiva. Con $\lambda$ diminuto tenemos el grado 12 desbocado; con $\lambda$ enorme, todos los coeficientes (salvo $w_0$) se van a cero y el modelo se vuelve casi una constante, con mucho sesgo. En medio, el polinomio de grado 12 regularizado compite con el mejor grado elegido a mano.
#
# La ventaja de $\lambda$ sobre el grado es que es un control **continuo**: no hay que decidir de antemano cuántos términos usar, solo cuánto castigar la complejidad.

# %%
fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, lam, titulo in zip(axs, [0.0, lam_opt], ["sin regularizar", f"ridge con λ = {lam_opt:.0e}"]):
    W = ajustar(X_ent, Y_ent, Q_RIDGE, lam)
    G = caracteristicas(xs, Q_RIDGE) @ W.T
    ax.plot(xs, G[:, :60], color=ROJO if lam == 0 else AZUL, alpha=0.18, lw=1)
    ax.plot(xs, G.mean(axis=1), color=ROJO if lam == 0 else AZUL, lw=3)
    ax.plot(xs, f(xs), color=rdp.estilo.TINTA, ls="--", lw=2)
    ax.set(ylim=(-2.2, 2.2), xlabel="$x$", title=f"Grado {Q_RIDGE}, {titulo}")
axs[0].set_ylabel("$y$")
plt.tight_layout()
plt.show()

# %% [markdown]
# Con la penalización, las 60 hipótesis de grado 12 se portan casi como las de grado 3 o 4: el promedio sigue cerca de $f$ y la dispersión se volvió pequeña.

# %% [markdown]
# ## 5. Datos reales: progresión de la diabetes
#
# ### Los datos y el problema
#
# **Datos.** Bradley Efron, Trevor Hastie, Iain Johnstone y Robert Tibshirani publicaron esta tabla en su artículo «Least Angle Regression» (*Annals of Statistics* 32, 2004), donde presentaron el algoritmo LARS que calcula de un golpe todo el camino de soluciones de lasso. Son **442 pacientes con diabetes**. Para cada uno se registraron, al inicio del estudio, diez variables de referencia:
#
# | variable | significado |
# |---|---|
# | `age`, `sex` | edad (años) y sexo (codificado 1 o 2) |
# | `bmi` | índice de masa corporal |
# | `bp` | presión arterial media |
# | `s1` … `s6` | seis mediciones en suero sanguíneo: colesterol total, LDL, HDL, cociente colesterol total/HDL, logaritmo de triglicéridos y glucosa |
#
# y, **un año después**, una medida cuantitativa de cuánto avanzó la enfermedad (valores entre 25 y 346; más alto es peor).
#
# **Problema.** Predecir la progresión a un año a partir de las mediciones iniciales, y entender **cuáles** de esas mediciones importan. Lo segundo es lo que motivó el artículo: un médico prefiere un modelo con pocas variables que pueda interpretar. La tabla viene incluida en scikit-learn; pedimos la versión en unidades originales (`scaled=False`) para estandarizar nosotros con los datos de entrenamiento.

# %%
diabetes = load_diabetes(as_frame=True, scaled=False)
Xd, yd = diabetes.data, diabetes.target
print(Xd.shape)
pd.concat([Xd, yd.rename("progresion")], axis=1).describe().round(1)

# %%
Xd_ent, Xd_pru, yd_ent, yd_pru = train_test_split(Xd, yd, test_size=0.25, random_state=SEMILLA)
escalador = StandardScaler().fit(Xd_ent)
Z_ent, Z_pru = escalador.transform(Xd_ent), escalador.transform(Xd_pru)
VARIABLES = list(Xd.columns)
print(f"entrenamiento: {len(yd_ent)} pacientes · prueba: {len(yd_pru)} pacientes")

# %% [markdown]
# **Una advertencia sobre $\lambda$ en scikit-learn.** Las dos clases no usan la misma escala:
#
# - `Ridge(alpha=λ)` minimiza $\lVert\mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 + \lambda\lVert\mathbf{w}\rVert_2^2$, igual que nosotros.
# - `Lasso(alpha=λ)` minimiza $\frac{1}{2N}\lVert\mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 + \lambda\lVert\mathbf{w}\rVert_1$, con el error **promediado**.
#
# Por eso los valores de $\lambda$ de uno y otro no son comparables entre sí. Ambos centran los datos y no penalizan el término constante.
#
# ### El camino de coeficientes
#
# Graficamos cómo cambia cada coeficiente cuando $\lambda$ va de enorme (todo encogido a cero) a diminuto (mínimos cuadrados). Para lasso usamos `lasso_path`, que calcula todo el camino de manera eficiente.

# %%
lam_r = np.logspace(-2, 5, 80)
camino_ridge = np.array([Ridge(alpha=a).fit(Z_ent, yd_ent).coef_ for a in lam_r])
lam_l, camino_lasso, _ = lasso_path(Z_ent, yd_ent - yd_ent.mean(), alphas=np.logspace(-2, 2, 80))
camino_lasso = camino_lasso.T

w_mc = LinearRegression().fit(Z_ent, yd_ent).coef_
DESTACADAS = {v: c for v, c in zip(pd.Series(np.abs(w_mc), VARIABLES).nlargest(4).index, [AZUL, ROJO, DORADO, rdp.estilo.PALETA["cafe"]])}

fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, lams, camino, nombre in zip(axs, [lam_r, lam_l], [camino_ridge, camino_lasso], ["Ridge", "Lasso"]):
    for j, v in enumerate(VARIABLES):
        ax.plot(lams, camino[:, j], color=DESTACADAS.get(v, "#AEB4C0"), lw=2.4 if v in DESTACADAS else 1.4)
    ax.axhline(0, color=rdp.estilo.TENUE, lw=0.8)
    ax.set(xscale="log", xlabel="$\\lambda$ (escala log)", title=f"{nombre}: coeficientes contra $\\lambda$")
    ax.invert_xaxis()
    x_etiq = lams.min() / 1.35
    ax.set_xlim(lams.max(), lams.min() / 8)
    i_min = np.argmin(lams)
    rdp.estilo.etiquetar(ax, x_etiq, {v: camino[i_min, j] for j, v in enumerate(VARIABLES)}, fontsize=10)
axs[0].set_ylabel("coeficiente (variables estandarizadas)")
plt.tight_layout()
plt.show()

# %% [markdown]
# El eje horizontal está invertido: a la izquierda $\lambda$ es enorme y todos los coeficientes valen cero; a la derecha $\lambda \to 0$ y llegamos a los coeficientes de mínimos cuadrados (las etiquetas). Las cuatro variables con coeficientes más grandes en mínimos cuadrados están en color.
#
# Los dos caminos se parecen en los extremos y son muy distintos en medio:
#
# - En **ridge** los coeficientes se encogen **suavemente** y todos a la vez; ninguno llega a cero antes de $\lambda = \infty$.
# - En **lasso** los coeficientes **entran uno por uno**: para cada $\lambda$ hay un subconjunto de variables con coeficiente exactamente cero. Las primeras en entrar (las más útiles) son `bmi` y `s5` (triglicéridos), y luego `bp`. Lasso hace **selección de variables**.
#
# También se nota algo curioso: `s1` y `s2` (colesterol total y LDL, muy correlacionadas) tienen en mínimos cuadrados coeficientes grandes y de signo contrario, que se cancelan. Es varianza pura: con variables casi colineales, los datos no pueden decidir cómo repartir el efecto. Ambas penalizaciones los encogen rápidamente.

# %% [markdown]
# ### Por qué lasso produce ceros: la geometría
#
# Lasso es ridge con otra norma: minimiza $\lVert\mathbf{y} - \mathbf{X}\mathbf{w}\rVert^2 + \lambda\lVert\mathbf{w}\rVert_1$, con $\lVert\mathbf{w}\rVert_1 = \sum_j |w_j|$. Por el mismo argumento de la sección 4, equivale a minimizar el error sujeto a $\lVert\mathbf{w}\rVert_1 \le C$. En dos dimensiones:
#
# - la región $\lVert\mathbf{w}\rVert_2^2 \le C$ es un **disco**;
# - la región $\lVert\mathbf{w}\rVert_1 \le C$ es un **rombo**, con esquinas sobre los ejes.
#
# Las curvas de nivel del error son elipses centradas en $\mathbf{w}_{\text{lin}}$. La solución es el primer punto donde una elipse, al crecer, toca la región. Un disco no tiene esquinas: el punto de contacto casi nunca está sobre un eje. El rombo sí: sus esquinas sobresalen, y es muy probable que la elipse las toque primero. En una esquina, una de las coordenadas vale **exactamente cero**.

# %%
A = np.array([[1.0, 0.55], [0.55, 0.6]])  # forma cuadrática del error (elipses inclinadas)
w_lin = np.array([1.6, 1.25])
C = 1.0


def error_cuad(w):
    d = w - w_lin
    return np.einsum("...i,ij,...j->...", d, A, d)


t = np.linspace(0, 2 * np.pi, 4000)
frontera_l2 = C * np.c_[np.cos(t), np.sin(t)]
frontera_l1 = C * np.c_[np.cos(t), np.sin(t)] / (np.abs(np.cos(t)) + np.abs(np.sin(t)))[:, None]

g1, g2 = np.meshgrid(np.linspace(-1.6, 3.0, 300), np.linspace(-1.6, 3.0, 300))
E = error_cuad(np.stack([g1, g2], axis=-1))

fig, axs = plt.subplots(1, 2, figsize=(12, 5.8))
for ax, frontera, nombre, color in zip(axs, [frontera_l2, frontera_l1], ["ridge: $\\|w\\|_2^2 \\leq C$", "lasso: $\\|w\\|_1 \\leq C$"], [AZUL, ROJO]):
    w_opt = frontera[np.argmin(error_cuad(frontera))]
    ax.add_patch(Polygon(frontera, closed=True, facecolor=color, alpha=0.15, edgecolor=color, lw=2))
    nivel = error_cuad(w_opt)
    ax.contour(g1, g2, E, levels=nivel * np.array([0.08, 0.3, 0.6, 1.0]), colors=[rdp.estilo.TENUE] * 3 + [rdp.estilo.TINTA],
               linewidths=[1, 1, 1, 2])
    ax.plot(*w_lin, "o", color=rdp.estilo.TINTA)
    ax.text(w_lin[0] + 0.08, w_lin[1] + 0.08, "$\\mathbf{w}_{\\mathrm{lin}}$")
    ax.plot(*w_opt, "o", color=color, ms=9, mec="white")
    ax.annotate(f"solución ({w_opt[0]:.2f}, {w_opt[1]:.2f})", w_opt, xytext=(-1.45, -1.35), color=color, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    ax.axhline(0, color=rdp.estilo.TENUE, lw=0.8)
    ax.axvline(0, color=rdp.estilo.TENUE, lw=0.8)
    ax.set(xlim=(-1.6, 3.0), ylim=(-1.6, 3.0), aspect="equal", xlabel="$w_1$", ylabel="$w_2$", title=nombre)
plt.tight_layout()
plt.show()

# %% [markdown]
# Mismo error, misma «cantidad» de restricción, distinto resultado: ridge da dos coeficientes chicos y distintos de cero; lasso pone $w_1 = 0$ exactamente.
#
# **El caso ortonormal, en fórmulas.** Si las columnas de $\mathbf{X}$ son ortonormales ($\mathbf{X}^\top\mathbf{X} = \mathbf{I}$), los problemas se separan por coordenada y se resuelven a mano. Con $\hat w_j$ el coeficiente de mínimos cuadrados:
#
# $$
# w_j^{\text{ridge}} = \frac{\hat w_j}{1 + \lambda},
# \qquad
# w_j^{\text{lasso}} = \operatorname{sign}(\hat w_j)\,\Big(|\hat w_j| - \frac{\lambda}{2}\Big)_+ .
# $$
#
# (Para lasso: en cada coordenada se minimiza $(w - \hat w)^2 + \lambda|w|$; si $w \ne 0$, la derivada da $w = \hat w - \tfrac{\lambda}{2}\operatorname{sign}(w)$, que solo es consistente si $|\hat w| > \lambda/2$; si no, el mínimo está en el punto donde la función no es derivable, $w = 0$.) Ridge **multiplica** todos los coeficientes por el mismo factor: ninguno llega a cero. Lasso **resta** una cantidad fija y corta en cero: es el *umbral suave* (*soft thresholding*). Los coeficientes pequeños desaparecen.

# %% [markdown]
# ### Elegir $\lambda$ con validación cruzada
#
# $\lambda$ no se puede elegir mirando $E_{\text{in}}$ (que siempre prefiere $\lambda = 0$) ni el conjunto de prueba (que debe quedar intacto hasta el final). Usamos **validación cruzada** dentro del conjunto de entrenamiento: se parte en 10 bloques, se entrena con 9 y se mide con el que queda, rotando, y se elige el $\lambda$ con menor error promedio. Lo estudiaremos a fondo en la libreta 3; por ahora basta con saber que `RidgeCV` y `LassoCV` lo hacen por nosotros.

# %%
pliegues = KFold(10, shuffle=True, random_state=SEMILLA)


def evaluar(modelo, X_e, X_p):
    modelo.fit(X_e, yd_ent)
    pred = modelo.predict(X_p)
    return np.sqrt(mean_squared_error(yd_pru, pred)), r2_score(yd_pru, pred)


mc = LinearRegression()
ridge = RidgeCV(alphas=np.logspace(-3, 4, 200), cv=pliegues)
lasso = LassoCV(alphas=np.logspace(-3, 2, 200), cv=pliegues, max_iter=50_000)
resultados = {nombre: evaluar(m, Z_ent, Z_pru) for nombre, m in
              [("mínimos cuadrados", mc), ("ridge", ridge), ("lasso", lasso)]}

print(f"λ elegido por validación cruzada:  ridge = {ridge.alpha_:.2f}   lasso = {lasso.alpha_:.3f}")
print(f"variables que conserva lasso: {np.sum(lasso.coef_ != 0)} de {len(VARIABLES)}; "
      f"descarta: {[v for v, c in zip(VARIABLES, lasso.coef_) if c == 0]}")
pd.DataFrame(resultados, index=["RECM prueba", "R² prueba"]).T.round(3)

# %%
pd.DataFrame({"mínimos cuadrados": w_mc, "ridge": ridge.coef_, "lasso": lasso.coef_}, index=VARIABLES).round(1)

# %% [markdown]
# Con estos datos **la regularización no mejora el error de prueba** (incluso lo empeora un poco, aunque la diferencia es mucho menor que lo que cambiaría con otra partición de los datos), y es una lección importante: con 331 pacientes y solo 10 variables, mínimos cuadrados tiene poca varianza (recuerda la brecha $\approx 2\sigma^2(d+1)/N$ de la sección 3), así que no hay mucho que ganar. Lo que sí cambia es el **modelo**: lasso llega al mismo error con menos variables, y ridge reparte de forma más sensata el efecto de `s1` y `s2`.
#
# La regularización brilla cuando el número de parámetros es grande frente a $N$. Hagámoslo así: agreguemos todos los productos de pares de variables y los cuadrados (las interacciones de grado 2). Pasamos de 10 a 65 variables con los mismos 331 pacientes.

# %%
expandir = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler()).fit(Z_ent)
Z2_ent, Z2_pru = expandir.transform(Z_ent), expandir.transform(Z_pru)
nombres2 = expandir[0].get_feature_names_out(VARIABLES)
print(f"variables: {Z2_ent.shape[1]}   rango de X: {np.linalg.matrix_rank(Z2_ent)}")

ridge2 = RidgeCV(alphas=np.logspace(-3, 4, 200), cv=pliegues)
lasso2 = LassoCV(alphas=np.logspace(-1, 2, 150), cv=pliegues, max_iter=100_000)
resultados2 = {nombre: evaluar(m, Z2_ent, Z2_pru) for nombre, m in
               [("mínimos cuadrados", LinearRegression()), ("ridge", ridge2), ("lasso", lasso2)]}
print(f"λ elegido:  ridge = {ridge2.alpha_:.1f}   lasso = {lasso2.alpha_:.3f}")
print(f"variables que conserva lasso: {np.sum(lasso2.coef_ != 0)} de {len(nombres2)}: "
      f"{[n for n, c in zip(nombres2, lasso2.coef_) if c != 0]}")
pd.DataFrame(resultados2, index=["RECM prueba", "R² prueba"]).T.round(3)

# %% [markdown]
# Ahora sí se ve la diferencia. Con 65 variables, mínimos cuadrados se ajusta al ruido de los 331 pacientes y empeora claramente en prueba; ridge y lasso, con los mismos 65 candidatos, regresan al nivel del modelo simple (lasso incluso lo mejora ligeramente) porque la penalización controla la varianza que agregaron las variables nuevas.
#
# Fíjate en el rango de la matriz: 64 y no 65. Como `sex` solo toma dos valores, `sex²` es una función lineal de `sex` y las dos columnas son colineales; $\mathbf{X}^\top\mathbf{X}$ es singular y mínimos cuadrados no tiene solución única (scikit-learn devuelve la de norma mínima). Para ridge no hay problema: $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ siempre es invertible. Lasso, en cambio, no sabe a cuál de las dos columnas idénticas darle el peso (aquí conservó ambas): con variables colineales, *cuáles* variables elige lasso es menos confiable que *cuántas*.

# %% [markdown]
# ## 6. Grados de libertad efectivos
#
# ### La pregunta
#
# Un polinomio de grado 12 sin regularizar tiene 13 parámetros libres. ¿Cuántos tiene «en realidad» si lo regularizamos? Intuitivamente, menos: no puede usarlos con libertad.
#
# ### La definición
#
# En regresión lineal (sin regularizar) las predicciones son $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ con la matriz sombrero $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$, que es una proyección sobre un espacio de dimensión $d$, y $\operatorname{tr}\mathbf{H} = d$: la traza cuenta los parámetros. Ridge también es lineal en $\mathbf{y}$, con $\mathbf{H}_\lambda = \mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top$, y por analogía se definen los **grados de libertad efectivos**
#
# $$
# \operatorname{df}(\lambda) = \operatorname{tr}\mathbf{H}_\lambda = \sum_{j=1}^{d} \frac{d_j^2}{d_j^2 + \lambda},
# $$
#
# donde la segunda igualdad sale de la expresión con valores singulares de la sección 4: $\mathbf{H}_\lambda = \sum_j \frac{d_j^2}{d_j^2+\lambda}\,\mathbf{u}_j\mathbf{u}_j^\top$ y $\operatorname{tr}(\mathbf{u}_j\mathbf{u}_j^\top) = 1$. Cada término está entre 0 y 1: una dirección con $d_j^2 \gg \lambda$ cuenta como un parámetro entero, una con $d_j^2 \ll \lambda$ casi no cuenta. $\operatorname{df}(0) = d$ (si $\mathbf{X}$ tiene rango completo) y $\operatorname{df}(\lambda) \to 0$ cuando $\lambda \to \infty$. Esta noción es la que retoman los criterios de selección de modelos como $C_p$ o AIC.
#
# (Para lasso no hay fórmula cerrada tan limpia, pero se demuestra que el número de coeficientes distintos de cero es un estimador insesgado de sus grados de libertad; Zou, Hastie y Tibshirani, 2007.)

# %%
def gl_efectivos(X, lams):
    d = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)  # ridge centra los datos
    return np.array([np.sum(d**2 / (d**2 + lam)) for lam in lams])


lams_df = np.logspace(-2, 6, 200)
df_10 = gl_efectivos(Z_ent, lams_df)
df_65 = gl_efectivos(Z2_ent, lams_df)
df_r, df_r2 = gl_efectivos(Z_ent, [ridge.alpha_])[0], gl_efectivos(Z2_ent, [ridge2.alpha_])[0]

# comprobación: la traza de la matriz sombrero coincide con la fórmula de los valores singulares
Xc = Z_ent - Z_ent.mean(axis=0)
H = Xc @ np.linalg.solve(Xc.T @ Xc + ridge.alpha_ * np.eye(Xc.shape[1]), Xc.T)
print(f"tr H_λ = {np.trace(H):.4f}   fórmula = {df_r:.4f}")
print(f"df en el λ de validación cruzada:  10 variables → {df_r:.2f}   65 variables → {df_r2:.2f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(lams_df, df_65, color=ROJO)
ax.plot(lams_df, df_10, color=AZUL)
for lam, df_val, color in [(ridge.alpha_, df_r, AZUL), (ridge2.alpha_, df_r2, ROJO)]:
    ax.plot(lam, df_val, "o", color=color, ms=9, mec="white")
    ax.annotate(f"λ de VC: df = {df_val:.1f}", (lam, df_val), xytext=(12, 8), textcoords="offset points", color=color, fontsize=11)
ax.set(xscale="log", xlabel="$\\lambda$ (escala log)", ylabel="grados de libertad efectivos",
       title="Ridge en los datos de diabetes: cuántos parámetros se usan de verdad", ylim=(0, 70))
ax.text(1.3e-2, 65, "65 variables (grado 2)", color=ROJO)
ax.text(1.3e-2, 13.5, "10 variables originales", color=AZUL)
plt.show()

# %% [markdown]
# Con las 10 variables originales, el $\lambda$ elegido deja unos 7.5 grados de libertad efectivos: un encogimiento moderado. Con las 65 variables de grado 2, la validación cruzada eligió un $\lambda$ que deja unos 30: es un modelo de 65 coeficientes que se comporta como uno de la mitad de tamaño. Así se reconcilian «muchos parámetros» y «buena generalización»: lo que importa no es cuántos parámetros hay, sino cuánta libertad tienen.

# %% [markdown]
# ## 7. Cierre
#
# - El error esperado fuera de la muestra es **exactamente** $\text{sesgo}^2 + \text{varianza} + \sigma^2$. El sesgo mide qué tan lejos queda la hipótesis promedio de $f$; la varianza, cuánto depende el resultado del conjunto de entrenamiento; $\sigma^2$ es un piso que nadie puede bajar.
# - Aumentar la complejidad baja el sesgo y sube la varianza; el mejor modelo **depende de $N$**. Con 15 puntos ganó el grado 3; con 100, el grado 9 ya es mejor que una recta.
# - Las **curvas de aprendizaje** distinguen los dos problemas: curvas juntas y altas indican sesgo (cambia de modelo); una brecha grande indica varianza (más datos o más regularización).
# - **Ridge** encoge los coeficientes; tiene solución cerrada $(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$, que existe aunque $\mathbf{X}^\top\mathbf{X}$ sea singular, y cambia un poco de sesgo por mucha menos varianza. Sus **grados de libertad efectivos** miden cuánta complejidad queda.
# - **Lasso** usa la norma $\ell_1$; las esquinas de su región ponen coeficientes exactamente en cero y hacen selección de variables.
# - Regularizar sirve sobre todo cuando hay muchos parámetros para pocos datos. $\lambda$ se elige con validación, nunca con el conjunto de prueba.

# %% [markdown]
# ## 8. Autoevaluación

# %%
rdp.quiz.mostrar("u03", ids=["u03-p06", "u03-p07", "u03-p08", "u03-p09"])

# %% [markdown]
# ## 9. Ejercicios
#
# 1. **Más datos.** Repite la tabla de la sección 2 con $N = 50$. ¿Cuál es ahora el grado óptimo? ¿Cambió más el sesgo o la varianza? Explica por qué.
# 2. **Más ruido.** Repite la sección 2 con $\sigma = 0.5$. ¿Se mueve el grado óptimo hacia arriba o hacia abajo? Justifica con la descomposición.
# 3. **Demostración.** Sea $\mathbf{X}$ de $N \times (d+1)$ con rango completo, $\mathbf{y} = \mathbf{X}\mathbf{w}^\ast + \boldsymbol{\varepsilon}$ y $\boldsymbol{\varepsilon}$ con media cero y covarianza $\sigma^2\mathbf{I}$. Demuestra que $\mathbb{E}\big[\lVert\mathbf{y} - \mathbf{H}\mathbf{y}\rVert^2\big] = \sigma^2\,(N - d - 1)$. Pista: $\mathbf{I} - \mathbf{H}$ es simétrica e idempotente, y $\operatorname{tr}(\mathbf{I} - \mathbf{H}) = N - d - 1$. Verifica numéricamente la fórmula $\mathbb{E}[E_{\text{in}}] = \sigma^2(1 - (d+1)/N)$ con la simulación de la sección 3 usando un $f$ que sí sea polinomio de grado 3.
# 4. **Sesgo y varianza de ridge.** Con $\mathbf{x}$ fijas y el mismo modelo del ejercicio anterior, demuestra que $\mathbb{E}[\mathbf{w}_{\text{ridge}}] = (\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}\,\mathbf{w}^\ast$ y que $\operatorname{Cov}(\mathbf{w}_{\text{ridge}}) = \sigma^2(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top\mathbf{X}+\lambda\mathbf{I})^{-1}$. Con valores singulares, muestra que la varianza de cada componente es menor que la de mínimos cuadrados.
# 5. **Umbral suave.** Completa la deducción del caso ortonormal de lasso, considerando por separado $w > 0$, $w < 0$ y $w = 0$ (en este último, usa que $0$ es mínimo si la derivada lateral derecha es $\ge 0$ y la izquierda $\le 0$).
# 6. **Lasso en la simulación.** Ajusta polinomios de grado 12 con `Lasso` (de scikit-learn, sobre las columnas $z, z^2, \dots, z^{12}$) a uno de los conjuntos de 15 puntos, para varios valores de `alpha`. ¿Qué potencias sobreviven? ¿Tiene sentido, dado que $\sin(2\pi x)$ es impar alrededor de $x = 1/2$?
# 7. **Red elástica.** `ElasticNetCV` combina ambas penalizaciones. Aplícalo a las 65 variables de diabetes con `l1_ratio=[0.1, 0.5, 0.9]`. ¿Mejora a ridge y a lasso? ¿Cuántas variables conserva?
