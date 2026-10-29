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
# # Entrenar una red con Fashion-MNIST
#
# **Unidad 6 · Ejemplo 2**
#
# En la libreta anterior construimos un perceptrón multicapa y calculamos su gradiente con retropropagación. Tener el gradiente no basta: hay que decidir **cómo usarlo** (cuánto avanzar, con cuántos datos a la vez, si recordar los pasos anteriores) y **cómo evitar que la red memorice** los datos de entrenamiento. Esas dos preguntas, optimización y regularización, ocupan la mayor parte del tiempo de quien entrena redes en la práctica. En esta libreta vamos a:
#
# 1. presentar **Fashion-MNIST** y escribir un bucle de entrenamiento en PyTorch, corto y reutilizable;
# 2. medir el **ruido del gradiente de un minilote** y verificar que su varianza baja como $1/B$;
# 3. ver qué hace la **tasa de aprendizaje** cuando es muy chica o muy grande;
# 4. analizar el **momento** en un problema cuadrático mal condicionado y derivar **Adam** con su corrección de sesgo; revisar los **calendarios** de la tasa y la diferencia entre **AdamW** y L2;
# 5. comparar SGD, SGD con momento y Adam en la misma red, con el mismo número de épocas;
# 6. regularizar con **parada temprana**, ***dropout*** y **normalización por lotes**, y comprobar que nada regulariza tanto como **tener más datos**.

# %% [markdown]
# ## 0. Preparación del entorno
#
# En Colab, la siguiente celda instala el paquete `rdp` del curso (estilo de gráficas, datos y cuestionarios). En tu máquina, si usas el entorno del repositorio, ya está instalado. Colab ya trae PyTorch y `torchvision`; si tiene GPU, la libreta la usa.

# %%
import os, sys

RAMA = "main"  # rama del repositorio del curso de la que se toman paquete y datos
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    # %pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@{RAMA}#subdirectory=paquete"
    pass

# %%
import copy, math, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torch.utils.data import TensorDataset, DataLoader

import rdp

rdp.estilo.usar()
AZUL, ROJO, DORADO_OSC, GRIS = rdp.estilo.AZUL, rdp.estilo.ROJO, rdp.estilo.DORADO_OSC, rdp.estilo.GRIS
TENUE = rdp.estilo.TENUE

SEMILLA = 2011
torch.manual_seed(SEMILLA)
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAIZ_DATOS = os.path.expanduser("~/.cache/rdp-datos")  # los datos de torchvision se guardan aquí, fuera del repositorio

# La verificación automática del repositorio usa menos épocas; las cifras del texto son las de la ejecución normal.
RAPIDO = os.environ.get("RDP_RAPIDO") == "1"
EPOCAS = 3 if RAPIDO else 30
print(f"dispositivo: {dispositivo} · épocas por corrida: {EPOCAS}")

# %% [markdown]
# ## 1. Los datos y el problema
#
# **Contexto.** Durante veinte años, MNIST (70,000 dígitos escritos a mano, de 28×28 píxeles) fue el conjunto de prueba por omisión en aprendizaje automático. En 2017, H. Xiao, K. Rasul y R. Vollgraf, investigadores de la tienda de ropa en línea Zalando, publicaron **Fashion-MNIST** («Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms», arXiv:1708.07747) con un argumento sencillo: MNIST es demasiado fácil (una red convolucional pasa de 99.7 % de exactitud), está sobreusado y no se parece a los problemas actuales de visión. Su propuesta conserva exactamente el formato de MNIST, para que se pueda usar sin cambiar una línea de código, pero con fotografías de prendas del catálogo de Zalando reducidas a 28×28 en escala de grises.
#
# **Datos.** 70,000 imágenes: 60,000 de entrenamiento y 10,000 de prueba, con 10 clases balanceadas: camiseta, pantalón, suéter, vestido, abrigo, sandalia, camisa, tenis, bolsa y botín. Cada imagen es un arreglo de $28 \times 28 = 784$ enteros entre 0 (fondo) y 255. Licencia MIT; `torchvision` los descarga la primera vez (unos 30 MB).
#
# **Problema.** Clasificar la prenda a partir de los 784 píxeles con un perceptrón multicapa, y estudiar cómo influyen el optimizador y la regularización en el resultado.
#
# **Partición.** Para que la libreta corra en unos minutos en CPU **y** para que el sobreajuste se vea con claridad, entrenaremos con **10,000** imágenes del conjunto de entrenamiento y usaremos otras **10,000** para validación (elegir épocas y comparar configuraciones). Las 10,000 de prueba oficiales solo se usan al final. En la sección 8 usaremos 50,000 imágenes de entrenamiento para ver qué cambia con más datos.

# %%
entren = torchvision.datasets.FashionMNIST(RAIZ_DATOS, train=True, download=True)
prueba = torchvision.datasets.FashionMNIST(RAIZ_DATOS, train=False, download=True)
CLASES = ["camiseta", "pantalón", "suéter", "vestido", "abrigo", "sandalia", "camisa", "tenis", "bolsa", "botín"]

X_todo = entren.data.float().div(255).reshape(-1, 784)   # 60,000 × 784, valores en [0, 1]
y_todo = entren.targets
X_prueba_crudo, y_prueba = prueba.data.float().div(255).reshape(-1, 784), prueba.targets

orden = torch.randperm(60000, generator=torch.Generator().manual_seed(SEMILLA))
i_ent, i_val, i_ent50 = orden[:10000], orden[50000:], orden[:50000]

# estandarizamos con la media y la desviación estándar de los píxeles del entrenamiento (un solo par de números)
media, desv = X_todo[i_ent].mean(), X_todo[i_ent].std()
estandarizar = lambda X: (X - media) / desv
X_ent, y_ent = estandarizar(X_todo[i_ent]), y_todo[i_ent]
X_val, y_val = estandarizar(X_todo[i_val]), y_todo[i_val]
X_prueba = estandarizar(X_prueba_crudo)
print(f"entrenamiento: {len(y_ent):,} · validación: {len(y_val):,} · prueba: {len(y_prueba):,}")
print("imágenes por clase en entrenamiento:", torch.bincount(y_ent).tolist())

# %%
fig, ejes = plt.subplots(10, 8, figsize=(8, 10.6))
for k in range(10):
    ejemplos = torch.nonzero(y_todo[i_ent] == k).flatten()[:8]
    for j, i in enumerate(ejemplos):
        ax = ejes[k, j]
        ax.imshow(X_todo[i_ent][i].reshape(28, 28), cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    ejes[k, 0].set_ylabel(CLASES[k], rotation=0, ha="right", va="center", fontsize=12)
fig.suptitle("Fashion-MNIST: ocho ejemplos de cada clase", y=0.995)
fig.tight_layout()
plt.show()

# %% [markdown]
# Algunas clases son fáciles (un pantalón no se parece a nada más), pero camiseta, camisa, suéter y abrigo comparten silueta, y una camisa se distingue de una camiseta por detalles (cuello, botones) que a 28×28 apenas se ven. Esa es la dificultad que MNIST no tenía.

# %% [markdown]
# ## 2. La red y el bucle de entrenamiento
#
# Usamos un perceptrón multicapa con dos capas ocultas de 512 y 256 neuronas y activación ReLU; la última capa da 10 puntajes (*logits*) y la pérdida es la entropía cruzada con *softmax*, como en la regresión logística multinomial de la Unidad 4. La opción `abandono` agrega *dropout* después de cada capa oculta (sección 6) y `normalizar`, normalización por lotes (sección 7).

# %%
def mlp(abandono=0.0, normalizar=False, ocultas=(512, 256)):
    capas, n = [], 784
    for k in ocultas:
        capas.append(nn.Linear(n, k))
        if normalizar:
            capas.append(nn.BatchNorm1d(k))
        capas.append(nn.ReLU())
        if abandono > 0:
            capas.append(nn.Dropout(abandono))
        n = k
    capas.append(nn.Linear(n, 10))
    return nn.Sequential(*capas)


print(mlp())
print(f"parámetros: {sum(p.numel() for p in mlp().parameters()):,}")

# %% [markdown]
# Son 535,818 parámetros para 10,000 imágenes de entrenamiento: la red puede memorizar los datos de sobra. Veremos que lo hace.
#
# El bucle de entrenamiento se escribe una vez y se reutiliza en toda la libreta. Una **época** es una pasada completa por los datos de entrenamiento, en **minilotes** de $B = 128$ imágenes en orden aleatorio; en cada minilote se calcula la pérdida, se retropropaga y el optimizador da un paso.

# %%
B = 128


def cargador(X, y, semilla=SEMILLA):
    """Minilotes de tamaño B en orden aleatorio (reproducible)."""
    return DataLoader(TensorDataset(X, y), batch_size=B, shuffle=True,
                      generator=torch.Generator().manual_seed(semilla))


def entrenar_epoca(modelo, minilotes, optimizador, calendario=None):
    modelo.train()                        # activa dropout y las estadísticas por minilote
    for xb, yb in minilotes:
        xb, yb = xb.to(dispositivo), yb.to(dispositivo)
        optimizador.zero_grad()
        perdida = F.cross_entropy(modelo(xb), yb)
        perdida.backward()
        optimizador.step()
        if calendario is not None:
            calendario.step()             # los calendarios de esta libreta cambian la tasa en cada paso


@torch.no_grad()
def evaluar(modelo, X, y):
    """Pérdida (entropía cruzada media) y exactitud en (X, y)."""
    modelo.eval()                         # desactiva dropout; la normalización usa sus promedios
    logits = torch.cat([modelo(X[i:i + 5000].to(dispositivo)).cpu() for i in range(0, len(X), 5000)])
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


# %% [markdown]
# Una función más arma una corrida completa: entrena `epocas` épocas, registra pérdida y exactitud en entrenamiento y validación al final de cada una, y **guarda una copia del modelo en la época de menor pérdida de validación** (la usaremos para la parada temprana). Al terminar evalúa en prueba los dos modelos: el de la última época y el guardado.

# %%
CORRIDAS = {}


def correr(nombre, modelo, optimizador, X=None, y=None, epocas=EPOCAS, calendario=None):
    X, y = (X_ent, y_ent) if X is None else (X, y)
    modelo.to(dispositivo)
    minilotes = cargador(X, y)
    historia, mejor, mejor_perdida = [], None, np.inf
    t0 = time.time()
    for e in range(1, epocas + 1):
        entrenar_epoca(modelo, minilotes, optimizador, calendario)
        pe, ae = evaluar(modelo, X, y)
        pv, av = evaluar(modelo, X_val, y_val)
        historia.append({"época": e, "pérdida ent.": pe, "exactitud ent.": ae, "pérdida val.": pv, "exactitud val.": av})
        if pv < mejor_perdida:
            mejor_perdida, mejor = pv, copy.deepcopy(modelo.state_dict())
    historia = pd.DataFrame(historia).set_index("época")
    final = evaluar(modelo, X_prueba, y_prueba)[1]
    e_mejor = int(historia["pérdida val."].idxmin())
    modelo.load_state_dict(mejor)
    temprana = evaluar(modelo, X_prueba, y_prueba)[1]
    CORRIDAS[nombre] = {"historia": historia, "modelo": modelo, "prueba final": final,
                        "prueba temprana": temprana, "época temprana": e_mejor}
    print(f"{nombre:<28} {time.time() - t0:5.1f} s · exactitud de prueba: {final:.3f} (última época), "
          f"{temprana:.3f} (época {e_mejor}, la de menor pérdida de validación)")
    return historia


# %% [markdown]
# Como punto de referencia, la red **sin capas ocultas** (`ocultas=()`): una sola capa lineal con *softmax*, es decir, la regresión logística multinomial de la Unidad 4, con 7,850 parámetros.

# %%
torch.manual_seed(SEMILLA)
lineal = mlp(ocultas=())
_ = correr("lineal (softmax)", lineal, torch.optim.Adam(lineal.parameters(), lr=1e-3))

# %% [markdown]
# ## 3. Descenso por gradiente estocástico
#
# ### El gradiente de un minilote
#
# La pérdida de entrenamiento es un promedio sobre los $N$ ejemplos, $E(\mathbf{w}) = \frac{1}{N}\sum_{n=1}^N \ell_n(\mathbf{w})$, así que su gradiente también lo es: $\mathbf{g} = \frac1N\sum_n \nabla\ell_n$. Calcularlo completo cuesta una pasada por todos los datos para dar **un** paso. El descenso por gradiente estocástico (SGD) usa en su lugar el gradiente de un minilote $\mathcal{B}$ de $B$ ejemplos elegidos al azar:
#
# $$\mathbf{g}_{\mathcal{B}} = \frac{1}{B}\sum_{n \in \mathcal{B}} \nabla\ell_n(\mathbf{w}), \qquad \mathbf{w} \leftarrow \mathbf{w} - \eta\,\mathbf{g}_{\mathcal{B}} .$$
#
# Si cada índice del minilote es uniforme sobre $\{1, \dots, N\}$, cada término tiene esperanza $\mathbf{g}$, así que $\mathbb{E}[\mathbf{g}_{\mathcal{B}}] = \mathbf{g}$: es un **estimador insesgado**. Y si los $B$ índices son independientes (con reemplazo), la varianza de un promedio de $B$ términos es la de uno entre $B$:
#
# $$\mathbb{E}\,\lVert\mathbf{g}_{\mathcal{B}} - \mathbf{g}\rVert^2 = \frac{\sigma^2}{B}, \qquad \sigma^2 = \frac1N\sum_n \lVert\nabla\ell_n - \mathbf{g}\rVert^2 .$$
#
# Sin reemplazo, que es como se toman los minilotes en la práctica, aparece el factor de población finita $\frac{N-B}{N-1}$ (ejercicio 1). Lo verificamos con la red después de una época de SGD: calculamos el gradiente completo sobre las 10,000 imágenes y lo comparamos con el de 200 minilotes de cada tamaño.

# %%
torch.manual_seed(SEMILLA)
red = mlp().to(dispositivo)
entrenar_epoca(red, cargador(X_ent, y_ent), torch.optim.SGD(red.parameters(), lr=0.05))


def gradiente(indices):
    red.zero_grad()
    F.cross_entropy(red(X_ent[indices].to(dispositivo)), y_ent[indices].to(dispositivo)).backward()
    return torch.cat([p.grad.flatten() for p in red.parameters()]).cpu()


g = gradiente(torch.arange(10000))
azar = torch.Generator().manual_seed(SEMILLA)
TAMANOS = [1, 4, 16, 64, 256, 1024]
ruido = []
for b in TAMANOS:
    d = [(gradiente(torch.randperm(10000, generator=azar)[:b]) - g).pow(2).sum().item() for _ in range(20 if RAPIDO else 200)]
    ruido.append(np.mean(d))
ruido = pd.DataFrame({"B": TAMANOS, "E‖g_B − g‖²": ruido})
ruido["B · E‖g_B − g‖²"] = ruido["B"] * ruido["E‖g_B − g‖²"]
print(f"‖g‖² = {g.pow(2).sum().item():.2f}")
ruido.round(3)

# %%
sigma2 = ruido["B · E‖g_B − g‖²"].iloc[:4].mean()
bb = np.logspace(0, np.log10(1024), 100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(bb, sigma2 / bb * (10000 - bb) / 9999, color=GRIS, lw=1.5, ls="--")
ax.loglog(ruido["B"], ruido["E‖g_B − g‖²"], "o", color=AZUL, ms=9, zorder=3)
ax.axhline(g.pow(2).sum().item(), color=ROJO, lw=1.5)
ax.text(1.1, g.pow(2).sum().item() * 1.25, "‖g‖²: el tamaño de la señal", color=rdp.estilo.TINTA)
ax.text(30, sigma2 / 30 * 1.6, "σ²/B · (N − B)/(N − 1)", color=TENUE)
ax.set_xlabel("tamaño del minilote B")
ax.set_ylabel("E‖g_B − g‖²  (ruido)")
ax.set_title("El ruido del gradiente de un minilote baja como 1/B")
plt.show()

# %% [markdown]
# $B \cdot \mathbb{E}\lVert\mathbf{g}_{\mathcal{B}} - \mathbf{g}\rVert^2$ es prácticamente constante, como predice la teoría (con $B = 1024$ baja un poco: es el factor $\frac{N-B}{N-1} \approx 0.9$). Lo notable es la escala: con **un** ejemplo, el ruido es unas 75 veces más grande que la señal $\lVert\mathbf{g}\rVert^2$; con $B = 128$ son del mismo orden. Aun así SGD funciona, porque los errores de pasos sucesivos son independientes y se promedian a lo largo de muchos pasos.
#
# El tamaño del minilote es un compromiso. Multiplicar $B$ por 4 cuesta 4 veces más por paso y solo reduce a la mitad la desviación estándar del ruido; en la práctica, con $B$ moderado se dan más pasos por el mismo cómputo y se avanza más rápido. Además, el ruido no es solo un costo: impide que el optimizador se asiente en mínimos muy estrechos y se ha observado que eso tiende a favorecer soluciones que generalizan mejor (Keskar et al., 2017). Es una forma de **regularización implícita**.
#
# ### La tasa de aprendizaje
#
# El parámetro más importante del entrenamiento es la tasa $\eta$. Entrenemos la misma red, desde la misma inicialización y con los mismos minilotes, con SGD y cinco tasas, durante tres épocas.

# %%
TASAS = [0.001, 0.01, 0.1, 0.5, 1.0]
barrido = {}
for eta in TASAS:
    torch.manual_seed(SEMILLA)
    red = mlp().to(dispositivo)
    opt = torch.optim.SGD(red.parameters(), lr=eta)
    perdidas = []
    for e in range(1 if RAPIDO else 3):
        red.train()
        for xb, yb in cargador(X_ent, y_ent, semilla=SEMILLA + e):
            opt.zero_grad()
            perdida = F.cross_entropy(red(xb.to(dispositivo)), yb.to(dispositivo))
            perdida.backward()
            opt.step()
            perdidas.append(perdida.item())
    barrido[eta] = np.array(perdidas)

fig, ax = plt.subplots(figsize=(10, 5))
estilos = {0.001: (GRIS, "-"), 0.01: (DORADO_OSC, "-"), 0.1: (AZUL, "-"), 0.5: (AZUL, ":"), 1.0: (ROJO, "-")}
finales = {}
for eta, p in barrido.items():
    color, linea = estilos[eta]
    if np.isfinite(p[-1]):
        suave = pd.Series(p).rolling(10, min_periods=1).mean()   # promedio móvil de 10 pasos para ver la tendencia
        ax.plot(suave, color=color, ls=linea, lw=2.2)
        finales[f"η = {eta}"] = suave.iloc[-1]
    else:
        k = int(np.argmax(~np.isfinite(p) | (p > 10)))           # primer paso en que la pérdida se dispara
        ax.plot(np.arange(k), p[:k], color=color, lw=2.2)
        ax.text(k + 3, 2.62, f"η = {eta}: diverge", color=color, fontsize=12, va="center")
rdp.estilo.etiquetar(ax, len(p) + 3, finales, separacion=0.13, fontsize=12)
ax.set_ylim(0, 2.8)
ax.set_xlim(0, len(p) * 1.15)
ax.set_xlabel("paso (minilote)")
ax.set_ylabel("pérdida del minilote (promedio móvil)")
ax.set_title("SGD con cinco tasas de aprendizaje, misma red y mismos minilotes")
plt.show()
for eta, p in barrido.items():
    if np.isfinite(p[-1]):
        print(f"η = {eta:<5}: pérdida media en los últimos 20 pasos = {np.mean(p[-20:]):.3f}")
    else:
        k = int(np.argmax(~np.isfinite(p)))
        print(f"η = {eta:<5}: diverge; pérdidas de los primeros pasos {np.round(p[:6], 2).tolist()}, NaN desde el paso {k + 1}")

# %% [markdown]
# Con $\eta = 0.001$ la pérdida apenas se mueve en 237 pasos; con $0.01$ baja, pero despacio. Entre $0.1$ y $0.5$ el entrenamiento es rápido; $0.5$ ya es más ruidoso. Con $\eta = 1$ la red **diverge**: en unos cuantos pasos los pesos se disparan y la pérdida se vuelve `NaN`. La regla práctica que sale de aquí, y que se usa en serio para elegir la tasa, es buscar la mayor tasa que todavía es estable y quedarse un poco por debajo.
#
# El porqué de la divergencia es el mismo que en el descenso por gradiente de la Unidad 4: cerca de un mínimo la pérdida es aproximadamente cuadrática, $E(\mathbf{w}) \approx E^* + \frac12(\mathbf{w}-\mathbf{w}^*)^\top H(\mathbf{w}-\mathbf{w}^*)$, y en la dirección del valor propio $\lambda$ de la hessiana el error se multiplica en cada paso por $1 - \eta\lambda$. Si $\eta > 2/\lambda_{\max}$, ese factor tiene valor absoluto mayor que 1 y el error crece exponencialmente.

# %% [markdown]
# ## 4. Momento, Adam y calendarios
#
# ### El zigzag y la bola pesada
#
# En la Unidad 4 vimos que cuando la hessiana está mal condicionada (número de condición $\kappa = \lambda_{\max}/\lambda_{\min}$ grande), el descenso por gradiente hace **zigzag**: la tasa debe ser chica para no divergir en la dirección más curva ($\eta < 2/\lambda_{\max}$), y con esa tasa se avanza con lentitud desesperante en la dirección más plana. La pérdida de una red, con cientos de miles de parámetros de escalas distintas, está típicamente muy mal condicionada.
#
# El **momento** (Polyak, 1964), o método de la bola pesada, agrega al paso una fracción $\beta$ del paso anterior:
#
# $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\,\nabla E(\mathbf{w}_t) + \beta\,(\mathbf{w}_t - \mathbf{w}_{t-1}) .$$
#
# En las direcciones donde el gradiente cambia de signo en cada paso (el zigzag), los pasos sucesivos se cancelan; en las que apunta siempre hacia el mismo lado, se acumulan, hasta una velocidad de $\eta/(1-\beta)$ veces el gradiente (10 veces con el valor usual $\beta = 0.9$).
#
# **Análisis en un cuadrático.** Sea $E(\mathbf{w}) = \frac12\mathbf{w}^\top H\mathbf{w}$, con valores propios entre $\lambda_{\min}$ y $\lambda_{\max}$. En la base de vectores propios las coordenadas se separan; en una dirección con valor propio $\lambda$ el error obedece $e_{t+1} = (1 + \beta - \eta\lambda)\,e_t - \beta\,e_{t-1}$, una recurrencia lineal con polinomio característico $r^2 - (1+\beta-\eta\lambda)\,r + \beta = 0$.
#
# - **Sin momento** ($\beta = 0$), $|r| = |1 - \eta\lambda|$. La mejor tasa iguala las dos direcciones extremas, $\eta = 2/(\lambda_{\min}+\lambda_{\max})$, y da la tasa de convergencia $\dfrac{\kappa - 1}{\kappa + 1}$.
# - **Con momento**, si las raíces son complejas su producto es $\beta$, así que $|r| = \sqrt\beta$ **para todas las direcciones a la vez**. Las raíces son complejas cuando $(1+\beta-\eta\lambda)^2 \le 4\beta$, es decir, cuando $(1-\sqrt\beta)^2 \le \eta\lambda \le (1+\sqrt\beta)^2$. Pidiendo que $[\lambda_{\min}, \lambda_{\max}]$ quepa justo en ese intervalo se obtiene
#
# $$\sqrt\beta = \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}, \qquad \eta = \frac{4}{(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}})^2}, \qquad \text{tasa de convergencia } \frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} .$$
#
# El número de iteraciones para reducir el error en un factor dado crece como $\kappa$ sin momento y como $\sqrt\kappa$ con él. Con $\kappa = 50$: $49/51 = 0.961$ contra $6.07/8.07 = 0.752$ por paso.
#
# El mismo argumento sirve si dejamos fija la tasa (por ejemplo, la mejor sin momento, $\eta = 2/51$) y solo agregamos momento: basta con que la dirección más plana tenga raíces complejas, $\beta \ge (1 - \sqrt{\eta\lambda_{\min}})^2 \approx 0.64$, y la tasa de convergencia es $\sqrt\beta \approx 0.80$. Con más momento la tasa $\sqrt\beta$ **empeora**: $\beta = 0.9$ da $0.95$. Comprobémoslo en un valle de $\kappa = 50$, empezando lejos del mínimo a lo largo del valle y un poco fuera del fondo.

# %%
kappa = 50.0
H = np.diag([1.0, kappa])                        # λ_min = 1 (a lo largo del valle), λ_max = 50 (a lo ancho)
w0 = np.array([-4.0, 0.5])


def bola_pesada(eta, beta, pasos=400):
    w, w_ant, camino = w0.copy(), w0.copy(), [w0.copy()]
    for _ in range(pasos):
        w, w_ant = w - eta * H @ w + beta * (w - w_ant), w
        camino.append(w.copy())
    return np.array(camino)


def iteraciones(camino, tol=1e-6):
    """Primera iteración en que ‖w_t − w*‖ / ‖w_0 − w*‖ < tol (el mínimo es w* = 0)."""
    error = np.linalg.norm(camino, axis=1) / np.linalg.norm(w0)
    return int(np.argmax(error < tol)) if (error < tol).any() else None


eta = 2 / (1 + kappa)                            # la mejor tasa sin momento
BETAS = [0.0, 0.5, 0.64, 0.9]
caminos = {b: bola_pesada(eta, b) for b in BETAS}
raiz = np.sqrt(kappa)
eta_opt, beta_opt = 4 / (raiz + 1) ** 2, ((raiz - 1) / (raiz + 1)) ** 2
optimo = bola_pesada(eta_opt, beta_opt)
for b, c in caminos.items():
    print(f"η = {eta:.4f}, β = {b:<4}: {iteraciones(c)} iteraciones para reducir el error a 10⁻⁶")
print(f"η = {eta_opt:.4f}, β = {beta_opt:.3f} (ambos óptimos): {iteraciones(optimo)} iteraciones")

# %%
fig, ejes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True)
U, V = np.meshgrid(np.linspace(-4.6, 1.2, 300), np.linspace(-0.75, 0.75, 200))
Z = 0.5 * (U**2 + kappa * V**2)
for ax, b, color in zip(ejes, [0.0, 0.64, 0.9], [ROJO, AZUL, DORADO_OSC]):
    c = caminos[b]
    ax.contour(U, V, Z, levels=[0.05, 0.3, 1, 2.5, 5, 8.5, 13], colors="#8DA3C9", linewidths=1.1)
    ax.plot(c[:40, 0], c[:40, 1], "-o", color=color, ms=3.5, lw=1.6)
    ax.plot(0, 0, "*", color=DORADO_OSC if color != DORADO_OSC else GRIS, ms=15, zorder=4)
    ax.set_aspect("equal"); ax.set_ylim(-0.75, 0.75)
    ax.set_title(("sin momento" if b == 0 else f"momento β = {b}") + f": primeros 40 pasos ({iteraciones(c)} iteraciones hasta 10⁻⁶)", loc="left", fontsize=13)
fig.suptitle(f"Un valle con κ = 50 y la misma tasa η = {eta:.3f}")
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 4.6))
for b, color in zip(BETAS, [ROJO, GRIS, AZUL, DORADO_OSC]):
    err = np.linalg.norm(caminos[b], axis=1) / np.linalg.norm(w0)
    ax.semilogy(err, color=color, lw=2.2, label=f"η = {eta:.3f}, " + ("sin momento" if b == 0 else f"β = {b}"))
ax.semilogy(np.linalg.norm(optimo, axis=1) / np.linalg.norm(w0), color=AZUL, lw=1.8, ls=":",
            label=f"η = {eta_opt:.3f}, β = {beta_opt:.2f} (óptimos)")
ax.legend(loc="upper right", fontsize=11)
ax.axhline(1e-6, color=TENUE, lw=0.8, ls="--")
ax.set_ylim(1e-9, 3); ax.set_xlim(0, 400)
ax.set_xlabel("iteración"); ax.set_ylabel("‖w_t − w*‖ / ‖w_0 − w*‖")
ax.set_title("Error relativo en el valle con κ = 50")
plt.show()

# %% [markdown]
# Sin momento, el camino rebota de una pared a la otra del valle y avanza poco a poco hacia el fondo. Con $\beta = 0.64$, las oscilaciones transversales se amortiguan en unos cuantos pasos y el avance a lo largo del valle se acelera: 79 iteraciones en lugar de 346, con **la misma tasa**. Con $\beta = 0.9$ el camino se pasa del mínimo y oscila a lo largo del valle (232 iteraciones): demasiado momento también es lento. Ajustando además la tasa, los valores óptimos llegan en 60 iteraciones (algo más de las 49 que da $0.752^t$, por un factor polinomial que aparece cuando las raíces son dobles en los extremos del espectro).
#
# En una red no conocemos $\lambda_{\min}$ ni $\lambda_{\max}$, y la pérdida no es cuadrática; se usa $\beta = 0.9$ y se ajusta $\eta$. Pero la misma recurrencia da una segunda ventaja, que sí se traslada directamente: **el límite de estabilidad**. La recurrencia $e_{t+1} = (1+\beta-\eta\lambda)e_t - \beta e_{t-1}$ converge si y solo si
#
# $$0 < \eta\lambda < 2(1+\beta) .$$
#
# (Sin momento, $\beta = 0$, es la condición $\eta\lambda < 2$ de antes.) En una dirección plana el momento avanza con el **paso efectivo** $\eta_{\text{ef}} = \eta/(1-\beta)$; escrita en términos de él, la condición es $\eta_{\text{ef}}\,\lambda < 2\,\frac{1+\beta}{1-\beta}$, que con $\beta = 0.9$ es $38$ en lugar de $2$. Es decir: **con el mismo avance en las direcciones planas, el momento tolera curvaturas 19 veces mayores** antes de volverse inestable. Lo veremos en la red en la sección 5.
#
# ### Adam y la corrección de sesgo
#
# El momento usa la misma tasa para todos los parámetros. **Adam** (Kingma y Ba, ICLR 2015) le da a cada parámetro su propia escala: lleva un promedio exponencial del gradiente (el momento, $\mathbf{m}$) y otro de su cuadrado ($\mathbf{v}$, coordenada a coordenada), y divide uno entre la raíz del otro:
#
# $$\mathbf{m}_t = \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t, \qquad \mathbf{v}_t = \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2,$$
#
# $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \qquad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}, \qquad \mathbf{w}_t = \mathbf{w}_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon},$$
#
# con valores usuales $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$ y $\eta = 10^{-3}$. Un parámetro con gradientes grandes da pasos de tamaño $\approx\eta$, igual que uno con gradientes diminutos: Adam es invariante a reescalar cada coordenada del gradiente.
#
# **¿De dónde sale la corrección?** Los promedios empiezan en $\mathbf{m}_0 = \mathbf{v}_0 = 0$. Desenrollando la recurrencia, $\mathbf{m}_t = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i}\mathbf{g}_i$. Si los gradientes tienen todos la misma esperanza $\mathbb{E}[\mathbf{g}]$,
#
# $$\mathbb{E}[\mathbf{m}_t] = (1-\beta_1)\sum_{i=1}^t\beta_1^{t-i}\,\mathbb{E}[\mathbf{g}] = (1-\beta_1^t)\,\mathbb{E}[\mathbf{g}],$$
#
# por la suma geométrica. El promedio está sesgado hacia cero por el factor $1-\beta_1^t$, y dividir entre él lo corrige; igual con $\mathbf{v}_t$ y $1-\beta_2^t$. Sin corrección, con un gradiente constante el paso sería $\eta\,\frac{1-\beta_1^t}{\sqrt{1-\beta_2^t}}$ en lugar de $\eta$: como $\beta_2$ está mucho más cerca de 1 que $\beta_1$, el denominador es el más sesgado y **los primeros pasos salen demasiado grandes**.

# %%
t = np.arange(1, 3001)
factor = (1 - 0.9**t) / np.sqrt(1 - 0.999**t)
print(f"paso sin corrección / paso corregido:  t = 1: {factor[0]:.2f}   máximo: {factor.max():.2f} (t = {t[factor.argmax()]})   "
      f"t = 100: {factor[99]:.2f}   t = 1000: {factor[999]:.2f}")
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.semilogx(t, factor, color=ROJO, lw=2.5)
ax.axhline(1, color=AZUL, lw=2)
ax.text(1.2, 1.25, "con corrección: 1", color=AZUL, fontsize=12)
ax.text(30, 5.4, "sin corrección: (1 − β₁ᵗ) / √(1 − β₂ᵗ)", color=ROJO, fontsize=12)
ax.set_xlabel("paso t"); ax.set_ylabel("tamaño del paso / η")
ax.set_title("Adam sin corrección de sesgo da pasos hasta 6.5 veces mayores al inicio")
plt.show()

# %% [markdown]
# ### Calendarios de la tasa de aprendizaje
#
# Una tasa constante es un compromiso: grande para avanzar al principio, chica para asentarse al final (con SGD, el ruido de los minilotes hace que el parámetro rebote alrededor del mínimo a una distancia proporcional a $\eta$). Un **calendario** cambia $\eta$ durante el entrenamiento. Los tres más usados:
#
# - **escalón**: multiplicar $\eta$ por 0.1 en épocas fijas (el de las redes residuales de 2015);
# - **coseno** (Loshchilov y Hutter, 2017): $\eta_t = \frac{\eta_0}{2}\big(1 + \cos(\pi t/T)\big)$, que baja suavemente de $\eta_0$ a 0 en $T$ pasos;
# - **calentamiento** (*warm-up*): subir $\eta$ linealmente desde casi 0 durante los primeros pasos, y luego seguir otro calendario. Al principio los pesos son aleatorios, los gradientes grandes y, con Adam, las estimaciones de $\mathbf{v}$ son pobres; el calentamiento evita que un paso enorme arruine la inicialización. Es indispensable en los *transformers* (sección 4 de la unidad).

# %%
pasos = np.arange(30 * 79)
T = len(pasos)
calendarios = {
    "constante": np.ones(T),
    "escalón": np.where(pasos < T / 2, 1.0, np.where(pasos < 0.8 * T, 0.1, 0.01)),
    "coseno": 0.5 * (1 + np.cos(np.pi * pasos / T)),
    "calentamiento + coseno": np.minimum(1, (pasos + 1) / (0.1 * T)) * 0.5 * (1 + np.cos(np.pi * pasos / T)),
}
fig, ax = plt.subplots(figsize=(9, 4.2))
for (nombre, c), color in zip(calendarios.items(), [GRIS, DORADO_OSC, AZUL, ROJO]):
    ax.plot(pasos / 79, c, color=color, lw=2.6 if nombre == "coseno" else 2.0, ls=":" if nombre == "calentamiento + coseno" else "-")
opciones = {"fontsize": 12, "color": rdp.estilo.TINTA}
ax.text(30.5, 1.0, "constante", va="center", **opciones)
ax.text(16.2, 0.14, "escalón", **opciones)
ax.text(13.5, 0.72, "coseno", **opciones)
ax.text(3.2, 0.35, "calentamiento\n+ coseno", **opciones)
ax.set_xlim(0, 34)
ax.set_xlabel("época"); ax.set_ylabel("η / η₀")
ax.set_title("Cuatro calendarios para la tasa de aprendizaje")
plt.show()

# %% [markdown]
# ### Decaimiento de pesos: con Adam, no es lo mismo que L2
#
# Con SGD, sumar a la pérdida la penalización L2 $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ de la Unidad 3 agrega $\lambda\mathbf{w}$ al gradiente, y el paso queda $\mathbf{w} \leftarrow (1 - \eta\lambda)\,\mathbf{w} - \eta\,\mathbf{g}$: los pesos «decaen» un factor fijo en cada paso. Por eso a la L2 se le llama **decaimiento de pesos** (*weight decay*).
#
# Con Adam la equivalencia se rompe. Si $\lambda\mathbf{w}$ se suma al gradiente, también pasa por la división entre $\sqrt{\hat{\mathbf{v}}}$, así que los pesos con gradientes grandes casi no decaen y los de gradientes chicos decaen mucho: la regularización depende de la historia de los gradientes. Loshchilov y Hutter (ICLR 2019) propusieron **desacoplarla**, aplicando el decaimiento directamente a los pesos:
#
# $$\text{AdamW:}\qquad \mathbf{w}_t = \mathbf{w}_{t-1} - \eta\Big(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\varepsilon} + \lambda\,\mathbf{w}_{t-1}\Big).$$
#
# En PyTorch, `torch.optim.Adam(..., weight_decay=λ)` es la versión acoplada (L2) y `torch.optim.AdamW(..., weight_decay=λ)` la desacoplada; AdamW es hoy el optimizador por omisión para *transformers*. El ejercicio 4 las compara en nuestra red.

# %% [markdown]
# ## 5. Tres optimizadores, la misma red
#
# Entrenamos la red de la sección 2 con tres optimizadores, desde la misma inicialización, con los mismos minilotes y durante 30 épocas (2,370 pasos):
#
# - SGD con $\eta = 0.05$;
# - SGD con momento, $\beta = 0.9$ y $\eta = 0.005$, para que el **paso efectivo** $\eta/(1-\beta) = 0.05$ sea el mismo que el de SGD;
# - Adam con su valor usual, $\eta = 10^{-3}$.

# %%
OPTIMIZADORES = {
    "SGD": lambda p: torch.optim.SGD(p, lr=0.05),
    "SGD + momento": lambda p: torch.optim.SGD(p, lr=0.005, momentum=0.9),
    "Adam": lambda p: torch.optim.Adam(p, lr=1e-3),
}
for nombre, crear in OPTIMIZADORES.items():
    torch.manual_seed(SEMILLA)
    red = mlp()
    correr(nombre, red, crear(red.parameters()))


# %%
def curvas(nombres, colores, titulo, columnas=("pérdida ent.", "pérdida val."), ylim=(0, 1.0)):
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, col in zip(ejes, columnas):
        fin = {}
        for n, c in zip(nombres, colores):
            h = CORRIDAS[n]["historia"][col]
            ax.plot(h.index, h.values, color=c, lw=2.2)
            fin[n] = h.values[-1]
        rdp.estilo.etiquetar(ax, len(h) + 0.8, fin, separacion=0.06 * (ylim[1] - ylim[0]))
        ax.set_xlim(0, len(h) * 1.4)
        ax.set_ylim(*ylim)
        ax.set_xlabel("época")
        ax.set_title(col.replace("ent.", "en entrenamiento").replace("val.", "en validación"))
    ejes[0].set_ylabel("entropía cruzada")
    fig.suptitle(titulo)
    fig.tight_layout()
    plt.show()


curvas(list(OPTIMIZADORES), [GRIS, AZUL, ROJO], "Mismas épocas, distintos optimizadores")
for n in OPTIMIZADORES:
    h = CORRIDAS[n]["historia"]["pérdida ent."]
    print(f"{n:<14} mayor aumento de la pérdida de entrenamiento entre dos épocas seguidas: {h.diff().max():.3f}")
filas = [i for i in (4, 9) if i < EPOCAS - 1] + [-1]  # épocas 5, 10 y la última
pd.DataFrame({n: CORRIDAS[n]["historia"].iloc[filas].stack() for n in OPTIMIZADORES}).unstack(0).round(3).T

# %% [markdown]
# Cuatro observaciones:
#
# 1. **SGD avanza a saltos.** Su pérdida de entrenamiento baja en promedio, pero de una época a otra llega a **subir** 0.5 (de 0.34 a 0.84 entre las épocas 25 y 26); la del momento nunca sube más de 0.02. Con el **mismo paso efectivo**, el momento avanza al mismo ritmo promedio (0.17 contra 0.18 en la época 30) y sin sobresaltos. Es lo que predice el análisis de estabilidad de la sección 4: con $\eta = 0.05$, SGD es inestable en cualquier dirección con curvatura $\lambda > 2/0.05 = 40$; con $\eta = 0.005$ y $\beta = 0.9$, el momento lo es solo si $\lambda > 2(1.9)/0.005 = 760$. Los saltos de SGD son consistentes con una red que, de vez en cuando, entra a una región donde alguna dirección es más curva que ese límite.
# 2. **Adam es el más rápido.** A la época 5 su pérdida de entrenamiento (0.30) es menor que la de momento (0.44) y la de SGD (0.61); en la época 30 llega a 0.05: prácticamente memorizó las 10,000 imágenes.
# 3. **Sobreajuste.** La pérdida de **validación** de Adam toca fondo en la época 6 (0.41) y luego **sube** hasta 0.7, aunque la de entrenamiento siga bajando. SGD y momento no han llegado a ese punto en 30 épocas: el sobreajuste llega antes con el optimizador que llega antes.
# 4. **La exactitud apenas se entera.** Mientras la pérdida de validación de Adam casi se duplica, su exactitud de validación sigue alrededor de 0.86 (termina en 0.868). La red acierta más o menos lo mismo, pero con una **confianza exagerada**: cuando se equivoca, asigna probabilidad casi cero a la clase correcta, y la entropía cruzada lo castiga. Si solo miráramos la exactitud no veríamos el problema, y unas probabilidades así de descalibradas (Unidad 3) no sirven para tomar decisiones.

# %% [markdown]
# ## 6. Regularización
#
# Todas las técnicas de esta sección buscan lo mismo que la regularización de la Unidad 3: restringir de alguna forma la complejidad efectiva del modelo para reducir la varianza a cambio de un poco de sesgo. La forma clásica, la penalización L2, ya la vimos como decaimiento de pesos. Las demás son propias de las redes.
#
# ### Parada temprana
#
# La más sencilla: vigilar la pérdida de validación y quedarse con el modelo de la época en que fue mínima. Es lo que hace `correr` al guardar una copia del modelo. Para un modelo lineal entrenado por descenso por gradiente desde $\mathbf{w} = 0$ se puede demostrar que detenerse en el paso $t$ equivale aproximadamente a una regularización L2 con $\lambda \approx 1/(\eta t)$: el número de pasos actúa como el inverso de la penalización (Goodfellow, Bengio y Courville, 2016, sección 7.8).

# %%
pd.DataFrame({n: {"época de menor pérdida val.": CORRIDAS[n]["época temprana"],
                  "pérdida val. mínima": CORRIDAS[n]["historia"]["pérdida val."].min(),
                  "pérdida val. final": CORRIDAS[n]["historia"]["pérdida val."].iloc[-1],
                  "exactitud prueba, parada temprana": CORRIDAS[n]["prueba temprana"],
                  "exactitud prueba, última época": CORRIDAS[n]["prueba final"]} for n in OPTIMIZADORES}).round(3)

# %% [markdown]
# Con SGD y momento la parada temprana no cambia nada: la mejor época es (casi) la última. Con Adam sí: el modelo de la época 6 tiene probabilidades razonables (pérdida de validación de 0.41 en vez de 0.7), pero **pierde 1.5 puntos de exactitud** de prueba (0.843 contra 0.858). En la época 6 la red todavía no había terminado de aprender a separar las clases, aunque ya empezaba a exagerar su confianza en los ejemplos de entrenamiento. La lección: **el criterio de parada debe ser la métrica que importa**. Si importa la exactitud, se vigila la exactitud de validación; si importan las probabilidades, la pérdida. Con *dropout*, como veremos, el dilema casi desaparece.
#
# La parada temprana tiene además un costo escondido: la validación se usa para elegir la época, así que su pérdida mínima es una estimación optimista. Por eso la comparación final se hace en prueba.
#
# ### *Dropout*
#
# El abandono, o *dropout* (Srivastava, Hinton, Krizhevsky, Sutskever y Salakhutdinov, *JMLR* 15, 2014), apaga al azar cada neurona oculta con probabilidad $p$ en cada paso de entrenamiento. Si $\mathbf{a}$ son las activaciones de una capa y $\mathbf{r}$ un vector de variables Bernoulli con $P(r_j = 1) = 1 - p$, en entrenamiento la capa siguiente recibe
#
# $$\tilde{\mathbf{a}} = \frac{\mathbf{r}\odot\mathbf{a}}{1-p}, \qquad \mathbb{E}[\tilde{\mathbf{a}}] = \mathbf{a},$$
#
# y en evaluación recibe $\mathbf{a}$ sin cambios. Dividir entre $1-p$ al entrenar (*dropout* **invertido**, el que implementa PyTorch) hace que la escala esperada sea la misma en los dos modos, y así no hay que tocar los pesos al predecir.

# %%
capa = nn.Dropout(p=0.5)
capa.train()
torch.manual_seed(SEMILLA)
salida = capa(torch.ones(100_000))
print("valores distintos:", salida.unique().tolist(), "· proporción de ceros:", (salida == 0).float().mean().item(),
      "· media:", round(salida.mean().item(), 4))
capa.eval()
print("en evaluación:", capa(torch.ones(5)).tolist())

# %% [markdown]
# **¿Por qué regulariza?** Dos lecturas complementarias:
#
# - **Coadaptación.** Ninguna neurona puede contar con que otra específica esté presente, así que cada una tiene que ser útil por sí sola, en compañía de subconjuntos aleatorios de las demás. Eso impide que se formen combinaciones frágiles que solo explican el ruido del entrenamiento.
# - **Ensamble.** Cada máscara $\mathbf{r}$ define una subred; con $n$ neuronas hay $2^n$ subredes que **comparten pesos**, y cada paso entrena una de ellas. Al evaluar sin *dropout* se usa una aproximación barata al promedio de todas (exacta para una capa lineal, aproximada después de las no linealidades). Promediar modelos reduce varianza, como veremos con los bosques aleatorios (Unidad 5).
#
# Repetimos las tres corridas con $p = 0.5$ después de cada capa oculta.

# %%
for nombre, crear in OPTIMIZADORES.items():
    torch.manual_seed(SEMILLA)
    red = mlp(abandono=0.5)
    correr(nombre + " + dropout", red, crear(red.parameters()))

# %%
fig, ejes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, nombre in zip(ejes, OPTIMIZADORES):
    for sufijo, color in [("", ROJO), (" + dropout", AZUL)]:
        h = CORRIDAS[nombre + sufijo]["historia"]
        ax.plot(h.index, h["pérdida ent."], color=color, lw=1.8, ls="--")
        ax.plot(h.index, h["pérdida val."], color=color, lw=2.4)
    ax.set_title(nombre)
    ax.set_xlabel("época")
    ax.set_ylim(0, 1.05)
ejes[0].set_ylabel("entropía cruzada")
ejes[2].plot([], [], color=ROJO, lw=2.4, label="sin dropout")
ejes[2].plot([], [], color=AZUL, lw=2.4, label="con dropout (p = 0.5)")
ejes[2].plot([], [], color=TENUE, lw=1.8, ls="--", label="entrenamiento")
ejes[2].plot([], [], color=TENUE, lw=2.4, label="validación")
ejes[2].legend(loc="upper left", fontsize=10.5, ncol=2)
fig.suptitle("Dropout: la brecha entre entrenamiento y validación se cierra")
fig.tight_layout()
plt.show()

# %% [markdown]
# Con *dropout*, la pérdida de entrenamiento baja más despacio (cada paso entrena una red «dañada») y, sobre todo, la de validación **deja de subir**: con Adam termina en 0.42 en lugar de 0.7. La brecha entre las dos curvas, que es nuestra estimación de la varianza, se reduce mucho. Con SGD y con momento, que aún no sobreajustaban, el efecto es menor, aunque las dos alcanzan pérdidas de validación más bajas (0.39 y 0.38). En exactitud de prueba la ganancia es pequeña; volveremos a eso en la tabla final.
#
# **Verificación de la interpretación como ensamble.** Si de verdad la red sin *dropout* aproxima el promedio de las subredes, promediar las probabilidades de muchas subredes elegidas al azar (dejando el *dropout* encendido al predecir) debería dar casi lo mismo.

# %%
red = CORRIDAS["Adam + dropout"]["modelo"]
with torch.no_grad():
    red.eval()
    p_det = F.softmax(red(X_val.to(dispositivo)), 1).cpu()
    red.train()                                       # dropout encendido: cada pasada es una subred distinta
    torch.manual_seed(SEMILLA)
    p_ens = torch.stack([F.softmax(red(X_val.to(dispositivo)), 1).cpu() for _ in range(100)]).mean(0)
    red.eval()
print(f"exactitud de validación, red completa:              {(p_det.argmax(1) == y_val).float().mean():.4f}")
print(f"exactitud de validación, promedio de 100 subredes:   {(p_ens.argmax(1) == y_val).float().mean():.4f}")
print(f"coinciden en la clase predicha en {(p_det.argmax(1) == p_ens.argmax(1)).float().mean():.1%} de las imágenes")

# %% [markdown]
# ## 7. Normalización por lotes
#
# La **normalización por lotes** (*batch normalization*; Ioffe y Szegedy, ICML 2015) estandariza cada neurona con la media y la varianza del minilote, y luego le devuelve una escala y un desplazamiento aprendidos:
#
# $$\hat z_j = \frac{z_j - \mu_{\mathcal{B},j}}{\sqrt{\sigma^2_{\mathcal{B},j} + \varepsilon}}, \qquad \text{salida}_j = \gamma_j\,\hat z_j + \beta_j .$$
#
# Al evaluar no hay minilote: se usan promedios móviles de $\mu$ y $\sigma^2$ acumulados en el entrenamiento (por eso hay que llamar a `modelo.eval()`). Su efecto principal es sobre la **optimización**: mantiene las entradas de cada capa en una escala estable, lo que permite tasas más grandes y hace el entrenamiento menos sensible a la inicialización. La explicación original («reduce el desplazamiento de covariables interno») se ha discutido mucho; Santurkar et al. (NeurIPS 2018) argumentan que lo que hace es suavizar el paisaje de la pérdida. Como la media y la varianza de cada minilote son ruidosas, también regulariza un poco.

# %%
torch.manual_seed(SEMILLA)
red = mlp(normalizar=True)
_ = correr("Adam + normalización", red, torch.optim.Adam(red.parameters(), lr=1e-3))
pd.DataFrame({n: CORRIDAS[n]["historia"].loc[[e for e in (1, 3, 5) if e <= EPOCAS], "pérdida ent."] for n in ["Adam", "Adam + normalización"]}).round(3)

# %% [markdown]
# Con normalización por lotes, la pérdida de entrenamiento baja más rápido en las primeras épocas y la pérdida de validación alcanza un mínimo más bajo y antes. Pero sin otra regularización, la red termina memorizando igual.
#
# ### Aumento de datos
#
# Otra forma de regularizar es **fabricar** ejemplos de entrenamiento con transformaciones que no cambian la clase: reflejar horizontalmente una prenda, desplazarla un par de píxeles, cambiar un poco el brillo. Es la manera de decirle al modelo qué invariancias esperamos sin cambiar su arquitectura, y es una de las técnicas más eficaces en imágenes (las redes convolucionales de la próxima libreta la usan siempre). El ejercicio 5 la prueba aquí.
#
# ## 8. Lo que más regulariza: más datos
#
# Para terminar, juntamos lo aprendido (Adam, *dropout*, un calendario de calentamiento + coseno y parada temprana) y entrenamos con **50,000** imágenes en lugar de 10,000, durante 15 épocas.

# %%
EPOCAS_50 = 2 if RAPIDO else 15
X_ent50, y_ent50 = estandarizar(X_todo[i_ent50]), y_todo[i_ent50]
pasos_totales = EPOCAS_50 * math.ceil(len(y_ent50) / B)
calentar = int(0.1 * pasos_totales)
torch.manual_seed(SEMILLA)
red = mlp(abandono=0.5)
opt = torch.optim.Adam(red.parameters(), lr=1e-3)
calendario = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda t: min(1.0, (t + 1) / calentar) * 0.5 * (1 + math.cos(math.pi * t / pasos_totales)))
_ = correr("Adam + dropout, 50,000 imágenes", red, opt, X_ent50, y_ent50, epocas=EPOCAS_50, calendario=calendario)

# %% [markdown]
# ## 9. La tabla final
#
# Exactitud en los 10,000 ejemplos de prueba, que no se usaron para nada hasta aquí.

# %%
tabla = pd.DataFrame({n: {"exactitud de prueba (última época)": c["prueba final"],
                          "exactitud de prueba (parada temprana)": c["prueba temprana"],
                          "época elegida": c["época temprana"],
                          "pérdida val. mínima": c["historia"]["pérdida val."].min(),
                          "pérdida val. final": c["historia"]["pérdida val."].iloc[-1]}
                      for n, c in CORRIDAS.items()}).T
tabla.round(3)

# %% [markdown]
# Lo que dice la tabla:
#
# - **La red le gana al modelo lineal**: de 0.83 a 0.85–0.86 con los mismos 10,000 datos. Es una mejora real pero modesta: con 784 píxeles crudos, una buena parte de lo que se puede aprender ya lo aprende la regresión *softmax*.
# - **Con 10,000 imágenes, todos los optimizadores y regularizadores terminan entre 0.85 y 0.86** de exactitud en la última época. Lo que sí cambia mucho es **cómo se llega** (SGD a saltos, Adam en pocas épocas) y **la calidad de las probabilidades** (la pérdida de validación final va de 0.39 a 0.7).
# - **Con 50,000 imágenes la exactitud sube a 0.887**, más que cualquier técnica aplicada a las 10,000. Antes de buscar el regularizador perfecto, hay que preguntarse si se pueden conseguir más datos.
#
# Las diferencias de unas milésimas entre corridas no deben tomarse en serio: con otra semilla cambian. Para afirmar que un optimizador es mejor que otro habría que repetir cada corrida con varias semillas, como hicimos con la validación cruzada en la Unidad 3.
#
# ## 10. Cierre
#
# Entrenar una red es resolver dos problemas a la vez. El de **optimización**: llegar a una pérdida de entrenamiento baja en un tiempo razonable, para lo que importan la tasa de aprendizaje (la mayor estable), el momento o Adam (que cancelan el zigzag del mal condicionamiento) y los calendarios. Y el de **generalización**: que lo aprendido sirva fuera del entrenamiento, para lo que vigilamos la validación y usamos parada temprana, *dropout*, decaimiento de pesos, normalización, aumento de datos y, cuando se puede, más datos.
#
# Una advertencia final: las redes modernas tienen muchos más parámetros que datos y aun así generalizan, algo que la teoría de la Unidad 3 no predice por sí sola. Al aumentar el tamaño del modelo, el error de prueba puede bajar, subir cerca del punto en que el modelo apenas interpola los datos y **volver a bajar** (el «doble descenso»; Belkin, Hsu, Ma y Mandal, *PNAS* 116, 2019; Nakkiran et al., 2019). Entender por qué es una pregunta de investigación abierta.
#
# *Referencias:* H. Xiao, K. Rasul y R. Vollgraf, «Fashion-MNIST», arXiv:1708.07747 (2017). B. T. Polyak, «Some methods of speeding up the convergence of iteration methods», *USSR Comput. Math. and Math. Phys.* 4 (1964). D. P. Kingma y J. Ba, «Adam: a method for stochastic optimization», ICLR (2015). I. Loshchilov y F. Hutter, «SGDR», ICLR (2017) y «Decoupled weight decay regularization», ICLR (2019). N. Srivastava et al., «Dropout», *JMLR* 15 (2014). S. Ioffe y C. Szegedy, «Batch normalization», ICML (2015). N. S. Keskar et al., «On large-batch training for deep learning», ICLR (2017). I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*, MIT Press (2016), capítulos 7 y 8.

# %% [markdown]
# ## 11. Autoevaluación

# %%
rdp.quiz.mostrar("u06", ids=["u06-p05", "u06-p06", "u06-p07", "u06-p08"])

# %% [markdown]
# ## 12. Ejercicios
#
# 1. **Varianza sin reemplazo.** Sean $\mathbf{u}_n = \nabla\ell_n - \mathbf{g}$, con $\sum_n\mathbf{u}_n = 0$ y $\sigma^2 = \frac1N\sum_n\lVert\mathbf{u}_n\rVert^2$. Demuestra que si el minilote es una muestra **sin reemplazo** de tamaño $B$, entonces $\mathbb{E}\lVert\mathbf{g}_{\mathcal{B}} - \mathbf{g}\rVert^2 = \frac{\sigma^2}{B}\cdot\frac{N-B}{N-1}$. Sugerencia: escribe $\mathbf{g}_{\mathcal{B}} - \mathbf{g} = \frac1B\sum_n I_n\mathbf{u}_n$ con $I_n$ el indicador de que $n$ está en el minilote, y calcula $\mathbb{E}[I_nI_m]$ para $n \ne m$. Compara con la tabla de la sección 3.
# 2. **La bola pesada.** (a) Verifica que si las raíces de $r^2 - (1+\beta-\eta\lambda)r + \beta = 0$ son complejas, su módulo es $\sqrt\beta$. (b) Con el cuadrático de la sección 4 ($\kappa = 50$), usa el valor «práctico» $\beta = 0.9$ y busca la mejor $\eta$. ¿Cuántas iteraciones necesitas? ¿Qué pasa con el camino si $\beta = 0.99$?
# 3. **Adam y la escala.** (a) Demuestra que si se multiplica la pérdida por una constante $c > 0$, los pasos de Adam no cambian (salvo por $\varepsilon$), y los de SGD se multiplican por $c$. (b) Con un gradiente constante $g \ne 0$, ¿cuál es exactamente el paso de Adam con corrección? Interpreta: ¿qué controla $\eta$ en Adam, la velocidad o la distancia por paso?
# 4. **Adam con L2 contra AdamW.** Entrena la red sin *dropout* con `Adam(lr=1e-3, weight_decay=λ)` y con `AdamW(lr=1e-3, weight_decay=λ)` para varios valores de $\lambda$ (por ejemplo $10^{-4}, 10^{-3}, 10^{-2}$ para Adam y $10^{-2}, 10^{-1}, 1$ para AdamW). Grafica la pérdida de validación final contra $\lambda$. ¿Por qué los rangos útiles de $\lambda$ son tan distintos? Calcula la norma de los pesos de la primera capa al final en cada caso.
# 5. **Aumento de datos.** Modifica `entrenar_epoca` para que, con probabilidad 1/2, refleje horizontalmente cada imagen del minilote (`xb.view(-1, 28, 28).flip(2).view(-1, 784)`) y la desplace al azar uno o dos píxeles (`torch.roll`). Entrena Adam sin *dropout* con las 10,000 imágenes. ¿Qué pasa con la brecha entre entrenamiento y validación? ¿Tiene sentido reflejar verticalmente estas imágenes?
