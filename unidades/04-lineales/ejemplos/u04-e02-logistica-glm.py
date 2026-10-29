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
# # Regresión logística y modelos lineales generalizados: del *Challenger* a las bicicletas
#
# **Unidad 4 · Ejemplo 2**
#
# La regresión lineal supone que la respuesta puede tomar cualquier valor real y que el ruido alrededor de la recta es gaussiano y de varianza constante. Muchas respuestas no son así: «¿se dañó la junta, sí o no?» es un 0 o un 1; «¿cuántas bicicletas se rentaron en esta hora?» es un conteo, nunca negativo, cuya variabilidad crece con su tamaño. En esta libreta vamos a:
#
# 1. plantear con datos reales del transbordador *Challenger* un problema de probabilidad de falla, y ver cómo un análisis incompleto de esos datos contribuyó a una mala decisión;
# 2. construir la **regresión logística**, derivar su gradiente y su hessiana, y programar desde cero el **método de Newton** (IRLS), que resulta ser una sucesión de mínimos cuadrados ponderados;
# 3. interpretar el modelo con **razones de momios**, medir su incertidumbre con *bootstrap* y ver qué pasa cuando los datos son **separables**;
# 4. pasar a varias clases con *softmax*;
# 5. generalizar todo a la **familia exponencial** y a los **modelos lineales generalizados** (GLM), y usar la **regresión de Poisson** para predecir rentas de bicicletas sin predicciones negativas.

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

from sklearn.linear_model import LogisticRegression, LinearRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_poisson_deviance

import rdp

rdp.estilo.usar()
SEMILLA = 2011
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TENUE = rdp.estilo.TENUE

# %% [markdown]
# # Parte 1. Regresión logística: el *Challenger*
#
# ## 1. Los datos y el problema
#
# **Contexto.** El 28 de enero de 1986 el transbordador espacial *Challenger* se desintegró 73 segundos después del despegue; murieron sus siete tripulantes. La causa fue la falla de las **juntas tóricas** (*O-rings*) de uno de los cohetes de combustible sólido: anillos de hule que sellan las uniones entre los segmentos del cohete. Esa mañana la temperatura en Cabo Cañaveral era de unos 31 °F (−0.6 °C), muy por debajo de la de cualquier lanzamiento anterior: el más frío había sido de 53 °F (11.7 °C).
#
# La noche anterior, los ingenieros de Morton Thiokol, la empresa que fabricaba los cohetes, recomendaron no lanzar por el frío: temían que el hule, rígido a baja temperatura, no sellara bien. En la teleconferencia se discutió si había relación entre la temperatura y los daños a las juntas, y la recomendación se revirtió. La Comisión Rogers, que investigó el accidente (1986), señaló que el análisis que se discutió consideraba **solo los vuelos que habían tenido incidentes** en las juntas, y omitía los vuelos sin daño.
#
# **Datos.** Los 23 vuelos anteriores de los que se recuperaron los cohetes, tal como los analizaron Dalal, Fowlkes y Hoadley (*Journal of the American Statistical Association*, 1989). Por vuelo tenemos la temperatura al despegue (`temp_f`, °F; `temp_c`, °C), la presión de la prueba de fugas (`presion`, psi), el número de juntas (`juntas`, siempre 6) y cuántas de ellas mostraron erosión o fuga de gases (`danadas`).
#
# **Problema.** Estimar la **probabilidad de que haya daño en alguna junta** en función de la temperatura al despegue, y en particular a 31 °F. Trabajaremos en °F, como los datos originales.

# %%
ch = rdp.datos.cargar("challenger")
ch["dano"] = (ch["danadas"] > 0).astype(int)
print(f"{len(ch)} vuelos; {ch['dano'].sum()} con daño en alguna junta")
ch.head()

# %% [markdown]
# ### La lección de los ceros
#
# Hagamos las dos gráficas: la que miraron (solo los vuelos con incidentes) y la que debieron mirar (todos los vuelos).

# %%
con_dano = ch[ch.danadas > 0]
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
ejes[0].scatter(con_dano.temp_f, con_dano.danadas, s=70, color=ROJO, zorder=3)
ejes[0].set_title("Solo los vuelos con incidentes: ¿hay relación?")
ejes[1].scatter(ch.temp_f[ch.danadas == 0], ch.danadas[ch.danadas == 0], s=70, color=AZUL, zorder=3, label="sin daño")
ejes[1].scatter(con_dano.temp_f, con_dano.danadas, s=70, color=ROJO, zorder=3, label="con daño")
ejes[1].set_title("Todos los vuelos: los fríos son los que fallan")
for ax in ejes:
    ax.set_xlabel("temperatura al despegue (°F)")
    ax.set_xlim(25, 85)
    ax.axvline(31, color=TENUE, ls="--", lw=1.2)
    ax.text(32, 2.6, "31 °F\n28 ene 1986", color=TENUE, fontsize=10, va="top")
ejes[0].set_ylabel("juntas dañadas (de 6)")
ejes[0].set_yticks([0, 1, 2, 3])
ejes[0].set_ylim(-0.3, 2.8)
ejes[1].legend(loc="upper right")
fig.tight_layout()
plt.show()

# %% [markdown]
# A la izquierda, los siete vuelos con incidentes: hubo daños a 53 °F y también a 75 °F; vista así, la temperatura no parece importar. A la derecha aparecen los 16 vuelos **sin** daño, y todos fueron a 66 °F o más. De los cuatro vuelos por debajo de 65 °F, los cuatro tuvieron daño; de los 19 restantes, solo tres.
#
# La lección es general y la volveremos a encontrar: **los ceros son datos**. Seleccionar las observaciones según la respuesta (aquí, quedarse solo con las que tuvieron $y > 0$) destruye justamente la información que permite ver la relación entre $x$ y $y$. Un clasificador entrenado solo con positivos no puede aprender qué los distingue de los negativos.

# %% [markdown]
# ## 2. El modelo logístico
#
# Sea $y \in \{0, 1\}$ (1 = hubo daño) y $x$ la temperatura. En esta libreta usamos etiquetas $\{0,1\}$, que son las naturales para una variable de Bernoulli y las que usa scikit-learn (*Learning From Data* usa $\{-1,+1\}$; el modelo es el mismo, cambia la forma de escribir la pérdida). Queremos modelar $p(\mathbf{x}) = P(y = 1 \mid \mathbf{x})$.
#
# Una recta $\mathbf{w}^\top\mathbf{x}$ no sirve directamente: toma cualquier valor real, y una probabilidad vive en $[0,1]$. La idea es transformar la probabilidad a una escala que sí recorra toda la recta real:
#
# - los **momios** (*odds*) $\dfrac{p}{1-p}$ van de 0 a $\infty$ («3 a 1» quiere decir $p = 3/4$);
# - los **log-momios** (*log-odds* o *logit*) $\log\dfrac{p}{1-p}$ van de $-\infty$ a $\infty$.
#
# El modelo logístico supone que los log-momios son **lineales** en $\mathbf{x}$ (con $x_0 = 1$ para el término independiente):
#
# $$\log\frac{p(\mathbf{x})}{1-p(\mathbf{x})} = \mathbf{w}^\top\mathbf{x}
# \quad\Longleftrightarrow\quad
# p(\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x}), \qquad \sigma(z) = \frac{1}{1+e^{-z}} .$$
#
# La **sigmoide** $\sigma$ es la inversa del logit. Dos propiedades que usaremos: $1 - \sigma(z) = \sigma(-z)$ y $\sigma'(z) = \sigma(z)\,(1-\sigma(z))$.
#
# ### Verosimilitud y entropía cruzada
#
# Cada $y_n$ es una variable de Bernoulli con parámetro $p_n = \sigma(\mathbf{w}^\top\mathbf{x}_n)$, así que $P(y_n\mid\mathbf{x}_n) = p_n^{y_n}(1-p_n)^{1-y_n}$. Con vuelos independientes, la log-verosimilitud es
#
# $$\ell(\mathbf{w}) = \sum_{n=1}^N \Big[y_n \log p_n + (1-y_n)\log(1-p_n)\Big]
# = \sum_{n=1}^N \Big[y_n\, \mathbf{w}^\top\mathbf{x}_n - \log\big(1 + e^{\mathbf{w}^\top\mathbf{x}_n}\big)\Big],$$
#
# y maximizarla equivale a minimizar el error de entrenamiento $E_{\text{in}}(\mathbf{w}) = -\frac{1}{N}\ell(\mathbf{w})$, la **entropía cruzada** promedio entre las etiquetas y las probabilidades del modelo (la *log-loss* de la Unidad 3).
#
# ### Gradiente y hessiana
#
# Derivando el segundo renglón término a término, con $\frac{\partial}{\partial\mathbf{w}}\log(1+e^{\mathbf{w}^\top\mathbf{x}_n}) = \sigma(\mathbf{w}^\top\mathbf{x}_n)\,\mathbf{x}_n = p_n\mathbf{x}_n$:
#
# $$\nabla \ell(\mathbf{w}) = \sum_{n}(y_n - p_n)\,\mathbf{x}_n = X^\top(\mathbf{y} - \mathbf{p}).$$
#
# Es la misma forma que en mínimos cuadrados, $X^\top(\mathbf{y} - X\mathbf{w})$: «datos por residuo». Pero ahora $\mathbf{p}$ depende de $\mathbf{w}$ de forma no lineal y la ecuación $\nabla\ell = \mathbf{0}$ no tiene solución cerrada. Derivando otra vez, con $\partial p_n/\partial\mathbf{w} = p_n(1-p_n)\mathbf{x}_n$:
#
# $$\nabla^2 \ell(\mathbf{w}) = -\sum_n p_n(1-p_n)\,\mathbf{x}_n\mathbf{x}_n^\top = -X^\top S X,
# \qquad S = \mathrm{diag}\big(p_1(1-p_1), \dots, p_N(1-p_N)\big).$$
#
# **Por qué el problema es convexo.** Para cualquier $\mathbf{v}$, $\mathbf{v}^\top X^\top S X\mathbf{v} = \sum_n p_n(1-p_n)(\mathbf{x}_n^\top\mathbf{v})^2 \ge 0$, porque $0 < p_n < 1$. Entonces la hessiana de $E_{\text{in}}$ es semidefinida positiva en todo punto: $E_{\text{in}}$ es convexa (estrictamente, si $X$ tiene rango completo), cualquier mínimo local es global, y no hay que preocuparse por el punto de partida.

# %% [markdown]
# ## 3. El método de Newton, desde cero
#
# Newton aproxima $\ell$ con su desarrollo de Taylor de segundo orden alrededor del punto actual y salta al máximo de esa parábola:
#
# $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \big[\nabla^2\ell\big]^{-1}\nabla\ell
# = \mathbf{w}^{(t)} + (X^\top S X)^{-1} X^\top(\mathbf{y} - \mathbf{p}).$$
#
# Son unas cuantas líneas de NumPy. Para evaluar $\log(1 + e^z)$ sin desbordes usamos `np.logaddexp(0, z)`.

# %%
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def newton_logistica(X, y, iteraciones=10, w=None):
    """Máxima verosimilitud de la regresión logística por el método de Newton (IRLS)."""
    w = np.zeros(X.shape[1]) if w is None else w.copy()
    historia = []
    for t in range(iteraciones):
        z = X @ w
        p = sigmoide(z)
        historia.append([t, *w, np.sum(y * z - np.logaddexp(0, z))])  # log-verosimilitud
        gradiente = X.T @ (y - p)
        hessiana = X.T @ (X * (p * (1 - p))[:, None])                  # X^T S X sin formar S
        w = w + np.linalg.solve(hessiana, gradiente)
    return w, pd.DataFrame(historia, columns=["iteración", "w0", "w1", "log-verosimilitud"]).set_index("iteración")


x = ch["temp_f"].to_numpy(dtype=float)
y = ch["dano"].to_numpy()
X = np.column_stack([np.ones_like(x), x])   # x0 = 1

w, historia = newton_logistica(X, y, iteraciones=8)
historia.style.format({"w0": "{:.6f}", "w1": "{:.6f}", "log-verosimilitud": "{:.8f}"})

# %% [markdown]
# Partiendo de $\mathbf{w} = \mathbf{0}$ (que da $p_n = 0.5$ a todos), en cinco iteraciones la log-verosimilitud ya no cambia en ocho decimales. Ésa es la **convergencia cuadrática** de Newton: cerca de la solución, el número de cifras correctas aproximadamente se duplica en cada paso. El descenso por gradiente necesitaría miles de iteraciones con estos datos sin estandarizar.
#
# Comparemos con scikit-learn sin regularización (`C=np.inf`, es decir, $\lambda = 0$; por omisión `LogisticRegression` regulariza, como veremos en la sección 5):

# %%
sk = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(x[:, None], y)
print(f"Newton desde cero: w0 = {w[0]:.4f}, w1 = {w[1]:.5f}")
print(f"scikit-learn:      w0 = {sk.intercept_[0]:.4f}, w1 = {sk.coef_[0, 0]:.5f}")

# %% [markdown]
# Coinciden: $w_0 = 15.04$, $w_1 = -0.232$.
#
# ### Newton es mínimos cuadrados ponderados iterativos (IRLS)
#
# Reescribamos el paso de Newton. Multiplicando por $X^\top S X$:
#
# $$X^\top S X\,\mathbf{w}^{(t+1)} = X^\top S X\,\mathbf{w}^{(t)} + X^\top(\mathbf{y}-\mathbf{p}) = X^\top S\big(X\mathbf{w}^{(t)} + S^{-1}(\mathbf{y}-\mathbf{p})\big) = X^\top S\,\mathbf{z},$$
#
# con la **respuesta de trabajo** $\mathbf{z} = X\mathbf{w}^{(t)} + S^{-1}(\mathbf{y} - \mathbf{p})$. Ésas son exactamente las ecuaciones normales de **mínimos cuadrados ponderados**: $\mathbf{w}^{(t+1)} = \arg\min_{\mathbf{w}}\sum_n s_n\,(z_n - \mathbf{w}^\top\mathbf{x}_n)^2$, con pesos $s_n = p_n(1-p_n)$. Cada paso de Newton es una regresión lineal sobre una respuesta «linealizada» $z_n$ (el log-momio actual más una corrección por el residuo), donde pesan más los vuelos cuya probabilidad está cerca de 0.5 (los más inciertos) y casi nada los que el modelo ya da por seguros. Como los pesos y la respuesta cambian en cada paso, se llama **mínimos cuadrados ponderados iterativos** (*iteratively reweighted least squares*, IRLS).
#
# Verifiquemos que un paso de IRLS hecho con `LinearRegression` y pesos da el mismo resultado que un paso de Newton:

# %%
w_t = np.array([10.0, -0.15])                       # un punto cualquiera
p_t = sigmoide(X @ w_t)
s_t = p_t * (1 - p_t)
z_t = X @ w_t + (y - p_t) / s_t                     # respuesta de trabajo

paso_newton = w_t + np.linalg.solve(X.T @ (X * s_t[:, None]), X.T @ (y - p_t))
mcp = LinearRegression().fit(x[:, None], z_t, sample_weight=s_t)
print("paso de Newton:                  ", paso_newton.round(6))
print("mínimos cuadrados ponderados en z:", np.array([mcp.intercept_, mcp.coef_[0]]).round(6))

# %% [markdown]
# ## 4. Interpretación e incertidumbre
#
# ### Razón de momios
#
# Como los log-momios son lineales, aumentar $x$ en una unidad suma $w_1$ a los log-momios, es decir, **multiplica los momios por $e^{w_1}$**, sin importar desde dónde se parta. A ese factor se le llama **razón de momios** (*odds ratio*).

# %%
rm = np.exp(-w[1])
p31 = sigmoide(w[0] + w[1] * 31)
print(f"por cada °F menos, los momios de daño se multiplican por e^{{{-w[1]:.3f}}} = {rm:.2f}")
print(f"por cada 10 °F menos: × {rm**10:.1f}")
print(f"P(daño | 53 °F) = {sigmoide(w[0] + w[1] * 53):.3f}   (el vuelo más frío previo)")
print(f"P(daño | 31 °F) = {p31:.4f}")

# %% [markdown]
# Cada grado Fahrenheit menos multiplica los momios de daño por 1.26; diez grados menos, por casi 10. Y a 31 °F el modelo da $P(y = 1 \mid 31\,°\text{F}) = 0.9996$.
#
# Ojo con lo que dice y no dice ese número. Primero, «daño» aquí es erosión o fuga en alguna junta, no la destrucción del vehículo; pero el diseño suponía que las juntas no se dañaban, y el daño era justo el mecanismo de la falla. Segundo, y más importante: **es una extrapolación**. Ningún vuelo despegó por debajo de 53 °F; el valor a 31 °F lo dicta la forma sigmoide que supusimos, no los datos. Lo que sí dicen los datos con claridad es que el riesgo crece al bajar la temperatura y que 31 °F estaba muy lejos de toda la experiencia previa: razón suficiente para no lanzar.
#
# ### *Bootstrap* no paramétrico
#
# ¿Cuánto cambiaría la curva con otros 23 vuelos? Con *bootstrap* (Unidad 3) remuestreamos los vuelos con reemplazo, reajustamos el modelo y vemos cómo varía. Hay un detalle: con tan pocos datos, algunas réplicas resultan **separables** (todos los vuelos con daño más fríos que todos los vuelos sin daño). En esas réplicas el estimador de máxima verosimilitud **no existe**, como veremos en la sección 5: la curva tiende a un escalón. Las contamos y las dejamos aparte.

# %%
def separable_1d(x, y):
    """¿Hay un umbral de temperatura que separe perfectamente las dos clases?"""
    return x[y == 1].max() < x[y == 0].min() or x[y == 1].min() > x[y == 0].max()


rng = np.random.default_rng(0)
B = 1000
malla = np.linspace(25, 85, 241)
Xm = np.column_stack([np.ones_like(malla), malla])
curvas, p31_boot, n_separables = [], [], 0
for _ in range(B):
    i = rng.integers(0, len(x), len(x))
    xb, yb = x[i], y[i]
    if yb.min() == yb.max() or separable_1d(xb, yb):
        n_separables += 1
        continue
    wb, _ = newton_logistica(X[i], yb, iteraciones=30)
    curvas.append(sigmoide(Xm @ wb))
    p31_boot.append(sigmoide(wb[0] + wb[1] * 31))
curvas, p31_boot = np.array(curvas), np.array(p31_boot)

print(f"réplicas separables (sin estimador de MV): {n_separables} de {B}")
q05, q50, q95 = np.quantile(p31_boot, [0.05, 0.5, 0.95])
print(f"P(daño | 31 °F) en las {len(p31_boot)} réplicas restantes: mediana {q50:.4f}, intervalo del 90 % [{q05:.3f}, {q95:.4f}]")

# %% [markdown]
# Las réplicas separables tienen todos sus vuelos con daño por debajo de los vuelos sin daño, así que su «curva» límite vale 1 a 31 °F: si las incluyéramos, el intervalo solo se correría hacia arriba.

# %%
banda = np.quantile(curvas, [0.05, 0.95], axis=0)
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.axvspan(25, 53, color=rdp.estilo.LINEA, alpha=0.7, lw=0)
ax.text(26, 0.42, "sin datos:\nextrapolación", color=TENUE, fontsize=11)
ax.fill_between(malla, *banda, color=AZUL, alpha=0.18, lw=0)
ax.plot(malla, sigmoide(Xm @ w), color=AZUL)
sacudida = np.random.default_rng(1).uniform(-0.025, 0.025, len(y))
ax.scatter(x, y + sacudida, s=55, color=np.where(y == 1, ROJO, GRIS), zorder=3)
ax.axvline(31, color=TENUE, ls="--", lw=1.2)
ax.annotate(f"31 °F: {p31:.4f}", (31, p31), (36, 0.9), color=rdp.estilo.TINTA, fontsize=11,
            arrowprops=dict(arrowstyle="->", color=TENUE))
ax.text(70, 0.55, "máxima verosimilitud\n(banda: 90 % bootstrap)", color=AZUL, fontsize=11)
ax.set_xlabel("temperatura al despegue (°F)")
ax.set_ylabel("P(daño en alguna junta)")
ax.set_title("Probabilidad estimada de daño en las juntas")
ax.set_xlim(25, 85)
plt.show()

# %% [markdown]
# La banda es angosta donde hay datos (60–80 °F) y se abre debajo de 60 °F, aunque a 31 °F casi todas las curvas ya llegaron a 1: la sigmoide no tiene otro lugar a donde ir. Esa estrechez a 31 °F es en parte un artefacto de la forma del modelo; otra familia de curvas podría dar valores distintos. La conclusión robusta no es «0.9996», sino «muy alta, y basada en extrapolar 22 °F más allá de los datos».

# %% [markdown]
# ## 5. Separabilidad: cuando el estimador no existe
#
# Supongamos que existe $\mathbf{v}$ con $\mathbf{v}^\top\mathbf{x}_n > 0$ para todos los $y_n = 1$ y $\mathbf{v}^\top\mathbf{x}_n < 0$ para todos los $y_n = 0$: los datos son **linealmente separables**. Entonces, para $c > 0$, el modelo $\mathbf{w} = c\,\mathbf{v}$ da $p_n = \sigma(c\,\mathbf{v}^\top\mathbf{x}_n) \to y_n$ cuando $c \to \infty$, y cada término de $\ell$ tiende a $\log 1 = 0$. La log-verosimilitud es siempre negativa y se acerca a 0 tanto como queramos, pero **ningún** $\mathbf{w}$ finito la alcanza: el máximo no existe y cualquier algoritmo que lo persiga hará crecer $\|\mathbf{w}\|$ sin límite.
#
# Lo veremos quitando los tres vuelos que «rompen» la separación: los que tuvieron daño a 70, 70 y 75 °F. Quedan cuatro vuelos con daño (53 a 63 °F), todos más fríos que los 16 sin daño (66 °F o más).

# %%
quitar = ch.dano.eq(1) & ch.temp_f.ge(65)
xs, ys = x[~quitar], y[~quitar]
Xs = X[~quitar]
print("¿separables?", separable_1d(xs, ys), f"({len(xs)} vuelos)")

ws = np.zeros(2)
filas = []
for t in range(1, 26):
    p = sigmoide(Xs @ ws)
    s = np.clip(p * (1 - p), 1e-300, None)
    ws = ws + np.linalg.lstsq(Xs.T @ (Xs * s[:, None]), Xs.T @ (ys - p), rcond=None)[0]
    z = Xs @ ws
    filas.append([t, ws[0], ws[1], np.linalg.norm(ws), np.sum(ys * z - np.logaddexp(0, z)), -ws[0] / ws[1]])
sep = pd.DataFrame(filas, columns=["iteración", "w0", "w1", "‖w‖", "log-verosimilitud", "umbral (°F)"]).set_index("iteración")
sep.iloc[[0, 1, 2, 4, 9, 14, 19, 24]].style.format("{:.4g}")

# %% [markdown]
# La log-verosimilitud se acerca a 0, $\|\mathbf{w}\|$ crece sin freno (aproximadamente en forma lineal con las iteraciones: cada paso de Newton suma una cantidad casi constante) y lo único que se estabiliza es el **umbral** $-w_0/w_1$, la temperatura donde $p = 0.5$, que queda en el hueco entre 63 y 66 °F. La curva se vuelve un escalón:

# %%
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.4))
ejes[0].plot(sep.index, sep["‖w‖"], marker="o", ms=4, color=AZUL)
ejes[0].set_xlabel("iteración de Newton")
ejes[0].set_ylabel(r"$\|\mathbf{w}\|$")
ejes[0].set_title("Datos separables: los pesos crecen sin límite")
malla2 = np.linspace(45, 85, 400)
for t, color in zip([1, 3, 6, 25], [GRIS, DORADO_OSC, ROJO, AZUL]):
    w0, w1 = sep.loc[t, ["w0", "w1"]]
    ejes[1].plot(malla2, sigmoide(w0 + w1 * malla2), color=color, label=f"iteración {t}")
ejes[1].scatter(xs, ys, s=45, color=np.where(ys == 1, ROJO, GRIS), zorder=3)
ejes[1].set_xlabel("temperatura (°F)")
ejes[1].set_ylabel("P(daño)")
ejes[1].set_title("La sigmoide tiende a un escalón")
ejes[1].legend(loc="upper right", frameon=True)
fig.tight_layout()
plt.show()

# %% [markdown]
# En la práctica esto es más común de lo que parece: con muchas variables y pocos datos (por ejemplo, textos con miles de palabras), los datos casi siempre son separables. El remedio es **regularizar**: minimizar $E_{\text{in}}(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2$ (excluyendo el término independiente). La penalización crece sin límite con $\|\mathbf{w}\|$, así que el mínimo existe y es único (la función es ahora estrictamente convexa). Por eso `LogisticRegression` regulariza por omisión, con $C = 1/\lambda$ multiplicando a la pérdida:

# %%
for C in [0.1, 1, 10, 100]:
    m = LogisticRegression(C=C, max_iter=10_000).fit(xs[:, None], ys)
    print(f"C = {C:>5}: w0 = {m.intercept_[0]:7.2f}, w1 = {m.coef_[0, 0]:6.3f}, umbral = {-m.intercept_[0] / m.coef_[0, 0]:.1f} °F")

# %% [markdown]
# Con regularización siempre hay solución finita; al crecer $C$ (menos regularización) los pesos crecen y la curva se empina, pero sin escaparse al infinito. El valor de $C$ se elige, como cualquier hiperparámetro, por validación cruzada.

# %% [markdown]
# ## 6. Varias clases: *softmax*
#
# Con $K$ clases el modelo tiene un vector de pesos $\mathbf{w}_k$ por clase y convierte los $K$ puntajes lineales en probabilidades con la función ***softmax***:
#
# $$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top\mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^\top\mathbf{x}}}.$$
#
# Con $K = 2$ se recupera la logística: dividiendo arriba y abajo entre $e^{\mathbf{w}_1^\top\mathbf{x}}$ queda $\sigma\big((\mathbf{w}_2 - \mathbf{w}_1)^\top\mathbf{x}\big)$. Esto muestra también que sumar el mismo vector a todos los $\mathbf{w}_k$ no cambia nada: solo importan las diferencias (sin regularización hay que fijar una clase de referencia; con regularización se resuelve sola). La pérdida es otra vez la entropía cruzada, $-\sum_n \log P(y_n \mid \mathbf{x}_n)$, convexa, y su gradiente respecto de $\mathbf{w}_k$ tiene la forma conocida $\sum_n\big(\mathbb{1}[y_n = k] - P(y=k\mid\mathbf{x}_n)\big)\mathbf{x}_n$.
#
# Un ejemplo pequeño con los pingüinos de la Unidad 2: distinguir las tres especies con dos medidas del pico, su longitud y su profundidad. Una sola de las dos medidas no basta (Chinstrap y Gentoo tienen picos igual de largos; Adelie y Chinstrap, igual de profundos), pero juntas sí.

# %%
medidas = ["bill_length_mm", "bill_depth_mm"]
p = rdp.datos.cargar("pinguinos").dropna(subset=medidas)
Xp, yp = p[medidas].to_numpy(), p["species"].to_numpy()

softmax = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))   # multinomial por omisión
print(f"exactitud con validación cruzada (5 particiones): {cross_val_score(softmax, Xp, yp, cv=5).mean():.3f}")
softmax.fit(Xp, yp)
nuevo = pd.DataFrame([[45.0, 17.0]], columns=medidas)
pd.DataFrame(softmax.predict_proba(nuevo.to_numpy()), columns=softmax.classes_, index=["pico de 45 × 17 mm"]).round(3)

# %%
g1, g2 = np.meshgrid(np.linspace(31, 61, 300), np.linspace(12.5, 22, 300))
clases = list(softmax.classes_)
zona = np.searchsorted(clases, softmax.predict(np.column_stack([g1.ravel(), g2.ravel()]))).reshape(g1.shape)
colores = [AZUL, ROJO, DORADO_OSC]
fig, ax = plt.subplots(figsize=(8.5, 5.6))
ax.contourf(g1, g2, zona, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#C9D3EC", "#F4C7C3", "#F3E2B3"], alpha=0.7)
for k, (especie, color) in enumerate(zip(clases, colores)):
    sel = yp == especie
    ax.scatter(Xp[sel, 0], Xp[sel, 1], s=22, color=color, label=especie, zorder=3)
ax.set_xlabel("longitud del pico (mm)")
ax.set_ylabel("profundidad del pico (mm)")
ax.set_title("Regiones de decisión de la regresión softmax")
ax.legend(loc="lower left", frameon=True)
plt.show()

# %% [markdown]
# Las fronteras entre regiones son rectas: en la frontera entre $k$ y $j$ se cumple $\mathbf{w}_k^\top\mathbf{x} = \mathbf{w}_j^\top\mathbf{x}$, una ecuación lineal. La regresión logística, con dos clases o con muchas, es un **clasificador lineal**; lo que la distingue de otros clasificadores lineales (el perceptrón, la SVM de la libreta 4) es que además entrega probabilidades.

# %% [markdown]
# # Parte 2. Modelos lineales generalizados: las bicicletas
#
# ## 7. La familia exponencial
#
# La regresión lineal y la logística se parecen demasiado para ser coincidencia: en ambas el gradiente es $X^\top(\mathbf{y} - \text{predicción})$. La explicación es que las dos son casos de un mismo esquema.
#
# Una distribución pertenece a la **familia exponencial** (en su forma más simple, la que nos basta) si su densidad o función de masa se puede escribir
#
# $$p(y \mid \theta) = h(y)\,\exp\big(\theta\, y - A(\theta)\big),$$
#
# donde $\theta$ es el **parámetro natural** y $A(\theta)$, la función de partición logarítmica, es lo que hace que la distribución sume 1. Derivando $\int p(y\mid\theta)\,dy = 1$ respecto de $\theta$ se obtiene $\int (y - A'(\theta))\,p(y\mid\theta)\,dy = 0$, es decir,
#
# $$\mu = \mathbb{E}[y] = A'(\theta), \qquad \text{y derivando otra vez,}\qquad \mathrm{Var}(y) = A''(\theta) \ge 0 .$$
#
# Un **modelo lineal generalizado** (GLM) tiene tres ingredientes: (1) una distribución de la familia exponencial para $y \mid \mathbf{x}$; (2) un predictor lineal $\eta = \mathbf{w}^\top\mathbf{x}$; (3) una **función liga** $g$ que conecta la media con el predictor, $g(\mu) = \eta$. La **liga canónica** es la que hace $\theta = \eta$, es decir, $g = (A')^{-1}$.
#
# | Distribución | Respuesta | $\theta$ | $A(\theta)$ | $\mu = A'(\theta)$ | Liga canónica | Modelo |
# |---|---|---|---|---|---|---|
# | Normal ($\sigma = 1$) | real | $\mu$ | $\theta^2/2$ | $\theta$ | identidad, $\eta = \mu$ | regresión lineal |
# | Bernoulli | $\{0,1\}$ | $\log\frac{p}{1-p}$ | $\log(1+e^\theta)$ | $\sigma(\theta)$ | logit, $\eta = \log\frac{\mu}{1-\mu}$ | regresión logística |
# | Poisson | $\{0,1,2,\dots\}$ | $\log\lambda$ | $e^\theta$ | $e^\theta$ | logaritmo, $\eta = \log\mu$ | regresión de Poisson |
#
# Por ejemplo, para Poisson: $p(y\mid\lambda) = \dfrac{\lambda^y e^{-\lambda}}{y!} = \dfrac{1}{y!}\exp\big(y\log\lambda - \lambda\big)$, así que $h(y) = 1/y!$, $\theta = \log\lambda$ y $A(\theta) = e^\theta$; en efecto $A'(\theta) = A''(\theta) = \lambda$: media y varianza iguales.
#
# ### El resultado unificador
#
# Con liga canónica, $\theta_n = \mathbf{w}^\top\mathbf{x}_n$, y la log-verosimilitud es $\ell(\mathbf{w}) = \sum_n\big[\log h(y_n) + y_n\,\mathbf{w}^\top\mathbf{x}_n - A(\mathbf{w}^\top\mathbf{x}_n)\big]$. Derivando:
#
# $$\nabla\ell(\mathbf{w}) = \sum_n\big(y_n - A'(\mathbf{w}^\top\mathbf{x}_n)\big)\mathbf{x}_n = X^\top(\mathbf{y} - \boldsymbol\mu),
# \qquad
# \nabla^2\ell(\mathbf{w}) = -X^\top\,\mathrm{diag}\big(A''(\theta_n)\big)\,X = -X^\top\,\mathrm{diag}\big(\mathrm{Var}(y_n)\big)\,X .$$
#
# Para **cualquier** GLM con liga canónica: el gradiente es «datos por residuo», la log-verosimilitud es cóncava (las varianzas son no negativas) y Newton es IRLS con pesos iguales a las varianzas. La logística fue el caso $\mathrm{Var} = p(1-p)$; para Poisson los pesos son $\mu_n$. Una consecuencia directa: en el óptimo $X^\top\mathbf{y} = X^\top\boldsymbol\mu$; si el modelo tiene término independiente, **la suma de las predicciones en entrenamiento es igual a la suma de las observaciones** (salvo la tolerancia del optimizador), y lo mismo dentro de cada categoría codificada con *one-hot*. Lo verificaremos.

# %% [markdown]
# ## 8. Regresión de Poisson para las bicicletas
#
# **Los datos.** Son los de la libreta anterior: las rentas por hora del sistema Capital Bikeshare de Washington, D. C., en 2011 y 2012 (Fanaee-T y Gama, 2014; UCI). El problema es predecir `total`, el número de bicicletas rentadas en una hora, a partir de la hora, el tipo de día, el clima y el año. Usamos exactamente la misma preparación, la misma partición (días 1–19 de cada mes para entrenar, del 20 en adelante para probar) y los modelos M3 y M4 de la libreta anterior.
#
# **Por qué Poisson.** `total` es un **conteo**: entero, no negativo, y con una variabilidad que crece con la media (a las 3 de la mañana se rentan entre 0 y 20 bicicletas; a las 5 de la tarde de un día laboral, entre 200 y 900). La regresión lineal supone ruido normal de varianza constante y una media que puede ser cualquier número, incluso negativo. La regresión de Poisson supone $y \mid \mathbf{x} \sim \text{Poisson}(\mu)$ con $\log\mu = \mathbf{w}^\top\mathbf{x}$: la media es $e^{\mathbf{w}^\top\mathbf{x}} > 0$ siempre, y la varianza crece con ella.

# %%
b = rdp.datos.cargar("bicis_washington", parse_dates=["fecha"])
b["anio"] = b.fecha.dt.year - 2011          # 0 = 2011, 1 = 2012 (el sistema creció)
b["clima"] = b.clima.clip(upper=3)          # la categoría 4 tiene 3 horas: se une a la 3
b["hora_lab"] = b.hora.astype(str) + "_" + b.laboral.astype(str)
entrena = b[b.fecha.dt.day <= 19]           # días 1–19 de cada mes: 10,886 horas
prueba  = b[b.fecha.dt.day >= 20]           # días 20 en adelante: 6,493 horas
num = ["temp", "humedad", "viento"]
print(f"entrenamiento: {len(entrena):,} horas; prueba: {len(prueba):,} horas")

# %%
def modelo(categoricas, estimador):
    return make_pipeline(
        make_column_transformer((OneHotEncoder(drop="first"), categoricas), (StandardScaler(), num)),
        estimador)


escalera = {"M3": ["hora", "clima", "anio", "laboral"], "M4": ["hora_lab", "clima", "anio"]}
ajustados, filas = {}, []
for nombre, cat in escalera.items():
    for familia, estimador in [("lineal", LinearRegression()), ("Poisson", PoissonRegressor(alpha=0, max_iter=3000))]:
        m = modelo(cat, estimador).fit(entrena, entrena.total)
        pred = m.predict(prueba)
        ajustados[nombre, familia] = m
        filas.append({"modelo": nombre, "familia": familia,
                      "RECM": root_mean_squared_error(prueba.total, pred),
                      "devianza de Poisson media": mean_poisson_deviance(prueba.total, np.clip(pred, 0.5, None)),
                      "% predicciones negativas": 100 * (pred < 0).mean()})
pd.DataFrame(filas).set_index(["modelo", "familia"]).round(1)

# %% [markdown]
# La **devianza de Poisson** es la medida natural de error para conteos: $D = \frac{2}{N}\sum_n\big[y_n\log(y_n/\hat\mu_n) - (y_n - \hat\mu_n)\big]$, que es el doble de cuánta log-verosimilitud de Poisson se pierde respecto de un modelo que predijera cada $y_n$ exactamente. Penaliza los errores en proporción al tamaño esperado del conteo: equivocarse por 20 bicicletas a las 3 de la mañana es mucho más grave que a las 5 de la tarde. Requiere $\hat\mu > 0$, así que a las predicciones lineales las recortamos a 0.5 como mínimo (un favor que les hacemos).
#
# Con exactamente las mismas variables, cambiar la familia baja la RECM de prueba de 86.4 a 72.2 en M4 y la devianza a la mitad (40.0 contra 20.9), y desaparecen las predicciones negativas, que en el modelo lineal eran el 10 % de las horas. ¿De dónde viene la mejora, si los dos modelos usan el mismo $\mathbf{w}^\top\mathbf{x}$? De la escala: en la lineal los efectos se **suman** y en la de Poisson se **multiplican**, que es como se comportan las rentas. Un día de lluvia no quita el mismo número de bicicletas a las 3 de la mañana que a las 6 de la tarde: quita la misma **proporción**.
#
# Verifiquemos de paso la consecuencia del resultado unificador: en entrenamiento, la suma de las predicciones de Poisson es igual a la de las observaciones, en total y dentro de cada categoría.

# %%
m4p, m4l = ajustados["M4", "Poisson"], ajustados["M4", "lineal"]
mu_tr = m4p.predict(entrena)
print(f"total observado: {entrena.total.sum():,}   total predicho: {mu_tr.sum():,.1f}")
pd.DataFrame({"observado": entrena.groupby("clima").total.sum(),
              "predicho (Poisson)": pd.Series(mu_tr, index=entrena.index).groupby(entrena.clima).sum().round(1)})

# %% [markdown]
# ### Un día de prueba
#
# Veamos las 24 predicciones de M4 para un día laboral concreto del conjunto de prueba: el martes 29 de marzo de 2011, un día nublado de principios de primavera.

# %%
dia = prueba[prueba.fecha == "2011-03-29"].sort_values("hora")
fig, ax = plt.subplots(figsize=(10.5, 4.8))
ax.axhline(0, color=TENUE, lw=1)
ax.bar(dia.hora, dia.total, color=rdp.estilo.PALETA["azul-tenue"], width=0.8)
ax.plot(dia.hora, m4l.predict(dia), marker="o", ms=5, color=ROJO)
ax.plot(dia.hora, m4p.predict(dia), marker="o", ms=5, color=AZUL)
negativos = m4l.predict(dia) < 0
ax.scatter(dia.hora[negativos], m4l.predict(dia)[negativos], s=110, facecolors="none", edgecolors=ROJO, lw=1.5, zorder=4)
ax.text(23.4, m4l.predict(dia)[-1], "lineal", color=ROJO, va="center", fontsize=11)
ax.text(23.4, m4p.predict(dia)[-1] + 25, "Poisson", color=AZUL, va="center", fontsize=11)
ax.text(8.6, dia.total.iloc[8] + 10, "observado", color=rdp.estilo.PALETA["azul-medio"], fontsize=11, va="bottom")
ax.text(6.3, -75, "predicciones negativas", color=ROJO, fontsize=11, va="center")
ax.set_xticks(range(0, 24, 2))
ax.set_xlim(-0.7, 25.2)
ax.set_xlabel("hora del día")
ax.set_ylabel("bicicletas rentadas")
ax.set_title("Martes 29 de marzo de 2011 (prueba): modelo M4 lineal contra Poisson")
plt.show()
print(f"horas con predicción lineal negativa ese día: {negativos.sum()} de 24")

# %% [markdown]
# El modelo lineal le pone a cada hora un efecto fijo y le suma o resta, igual a todas, los efectos del clima, el año y la temperatura. En un día fresco y nublado de marzo de 2011 la resta es grande, y en la madrugada, cuando casi no hay rentas, deja la predicción por debajo de cero. Y para compensar en promedio, en las horas pico se pasa de largo. El modelo de Poisson aplica los mismos efectos como factores, y la forma del día (madrugada casi vacía, picos de entrada y salida del trabajo) se conserva a la escala que toca.
#
# ### Interpretación multiplicativa
#
# Como $\log\mu = \mathbf{w}^\top\mathbf{x}$, cambiar una variable que entra con coeficiente $w_j$ multiplica la media por $e^{w_j}$, igual que la razón de momios en la logística.

# %%
nombres = m4p[0].get_feature_names_out()
coef_p = pd.Series(m4p[-1].coef_, index=nombres)
coef_l = pd.Series(m4l[-1].coef_, index=nombres)
temp_de = m4p[0].named_transformers_["standardscaler"].scale_[0]
efectos = pd.DataFrame({
    "w (Poisson)": coef_p[["onehotencoder__clima_2", "onehotencoder__clima_3", "onehotencoder__anio_1", "standardscaler__temp"]],
    "factor e^w": np.exp(coef_p[["onehotencoder__clima_2", "onehotencoder__clima_3", "onehotencoder__anio_1", "standardscaler__temp"]]),
    "lineal (bicis/hora)": coef_l[["onehotencoder__clima_2", "onehotencoder__clima_3", "onehotencoder__anio_1", "standardscaler__temp"]],
})
efectos.index = ["nublado (vs. despejado)", "lluvia o nieve (vs. despejado)", "2012 (vs. 2011)", f"temperatura (+1 desv. est. = {temp_de:.1f} °C)"]
efectos.round(3)

# %% [markdown]
# En el modelo de Poisson, con todo lo demás igual:
#
# - la **lluvia o nieve** (clima 3) multiplica las rentas esperadas por $e^{-0.499} = 0.61$: un 39 % menos, a cualquier hora;
# - el **año 2012** las multiplica por $e^{0.472} = 1.60$: el sistema creció un 60 % de un año al otro;
# - 7.8 °C más de temperatura las multiplica por 1.27.
#
# El modelo lineal dice, en cambio, que la lluvia quita unas 72 bicicletas por hora: a las 3 de la mañana, cuando se rentan cinco, eso es absurdo, y es justo lo que produce los negativos de la gráfica. Del mismo modo, «2012 suma 87 bicicletas por hora» no tiene sentido de madrugada, mientras que «2012 multiplica por 1.6» sí.
#
# ## 9. Cierre: elegir la familia es elegir qué ruido se espera
#
# Logística y Poisson no son técnicas aisladas: son la regresión lineal con otra distribución para el ruido y una liga que respeta el rango de la respuesta. Elegir la familia es declarar qué tipo de variabilidad esperamos alrededor de la media: simétrica y constante (normal), de moneda cargada (Bernoulli), de conteo que crece con la media (Poisson). Esa elección fija la pérdida (cuadrados, entropía cruzada, devianza de Poisson), y a cambio obtenemos predicciones en el rango correcto, coeficientes con una interpretación natural (aditiva, razones de momios, factores multiplicativos) y, con la liga canónica, un problema convexo que Newton/IRLS resuelve en unas cuantas iteraciones.
#
# Y la lección del *Challenger* vale para cualquier modelo: antes de ajustar, preguntarse **qué datos faltan** y por qué.
#
# *Referencias:* Presidential Commission on the Space Shuttle Challenger Accident, *Report to the President* (1986). S. R. Dalal, E. B. Fowlkes y B. Hoadley, «Risk analysis of the space shuttle: pre-Challenger prediction of failure», *JASA* 84 (1989). P. McCullagh y J. A. Nelder, *Generalized Linear Models*, 2.ª ed., Chapman & Hall (1989).

# %% [markdown]
# ## 10. Autoevaluación

# %%
rdp.quiz.mostrar("u04", ids=["u04-p06", "u04-p07", "u04-p08", "u04-p09"])

# %% [markdown]
# ## 11. Ejercicios
#
# 1. **Logística con etiquetas $\pm 1$.** Demuestra que, con $y_n \in \{-1, +1\}$, la entropía cruzada se escribe $E_{\text{in}}(\mathbf{w}) = \frac{1}{N}\sum_n \log\big(1 + e^{-y_n\mathbf{w}^\top\mathbf{x}_n}\big)$, como en *Learning From Data*, y que su gradiente es $-\frac{1}{N}\sum_n y_n\,\mathbf{x}_n\,\sigma(-y_n\mathbf{w}^\top\mathbf{x}_n)$. ¿Qué vuelos «empujan» más el gradiente?
# 2. **Binomial en lugar de Bernoulli.** Cada vuelo tiene 6 juntas y sabemos cuántas se dañaron. Modela `danadas` como binomial$(6, p(x))$ con $p(x) = \sigma(w_0 + w_1 x)$: escribe la log-verosimilitud, muestra que el gradiente es $\sum_n (d_n - 6p_n)\mathbf{x}_n$ y adapta `newton_logistica`. Compara $P(\text{una junta dañada} \mid 31\,°\text{F})$ con la del modelo de la sección 3. (Ese fue el modelo de Dalal, Fowlkes y Hoadley.)
# 3. **Descenso por gradiente contra Newton.** Ajusta la logística del *Challenger* por descenso por gradiente con $x$ sin estandarizar y luego estandarizada. ¿Cuántas iteraciones necesitas en cada caso para llegar a 4 cifras de $\hat{\mathbf{w}}$? Explícalo con el número de condición de $X^\top S X$.
# 4. **Poisson desde cero.** Escribe IRLS para la regresión de Poisson (pesos $\mu_n$, respuesta de trabajo $z_n = \eta_n + (y_n - \mu_n)/\mu_n$) y úsalo con el modelo M3 de las bicicletas. Compara tus coeficientes con los de `PoissonRegressor(alpha=0)`. Sugerencia: empieza con $\mathbf{w}$ que dé $\mu_n = \bar y$.
# 5. **¿Es Poisson el ruido correcto?** Con el modelo M4 de Poisson, agrupa las horas de prueba por deciles de $\hat\mu$ y compara, en cada grupo, la media de $y$ con su varianza. ¿Se cumple $\mathrm{Var}(y) \approx \mu$? Si la varianza es mucho mayor (sobredispersión), ¿qué le pasa a los coeficientes y qué a la incertidumbre que reportaríamos? Investiga la familia binomial negativa.
