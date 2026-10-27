---
title: Examen rápido No. 2
author: Reconocimiento de Patrones (2023-2)
date: Julio Waissman Vilanova
---

\vspace{10mm}

**Nombre:** _______________________________________________________

\vspace{5mm}

# Problema 1

Consideremos una función conocida $f:[-1, 1] \to \mathbb{R}$ dada por $f(x) = \sin(\pi x)$ y con una muestra de datos de entrenamento tomada a partir de una distribución uniforme. Vamos a asumir que el conjunto de aprendizaje sólamente tiene dos datos tomados independientemente, y que el algoritmo de aprendizaje encuentra el valor que minimiza el error cuadrado medio (MSE) en los ejemplos.

Si consideramos una hipótesis del tipo:

$$
h(x) = ax
$$

\vspace{3mm}

- ¿Cual es la formula para obtener el valor óptimo $a^*$ con dos valores $\mathcal{D} = \{(x_1, y_1),(x_2, y_2)\}?$

\vspace{10mm}

- ¿Cual sería el valor de $$\bar{h}(x) = E_{\mathcal{D}\in \{ \mathcal{D}_1, \ldots, mathcal{D}_k \}}[h^\mathcal{D}]$$  

si se pone en función de $x$ (i.e. $\bar{h}(x) = \bar{a}x$):

\newpage

# Problema 2

Para el modelo considerado en el problema 1, encierra la hipótesis que crees que tendría el menor error fuera de muestra:

1. $h(x) = b$
2. $h(x) = ax$
3. $h(x) = ax + b$
4. $h(x) = ax^2$
5. $h(x) = ax^2 + b$

\vspace{3mm}

- Explica tus razones:

\vspace{3cm}


# Problema 3

Asumimos que tenemos un conjunto de $K$ posibles modelos a usar (o su combinación usando alguna técnica de *AutoML*) $\mathcal{H}_1, \ldots, \mathcal{H}_K$. Asumimos tambien que todas las hipótesis tienen una dimensión VC finita. 

- ¿Cuales de las siguientes aseveraciones son correctas, si decidimos usar todos los modelos (esto es, la union de todos los modelos)?:

1. $0 \leq d_{vc}(\bigcup_{k=1}^K \mathcal{H}_k) \leq \sum_{k=1}^K d_{vc}(\mathcal{H}_k)$
2. $0 \leq d_{vc}(\bigcup_{k=1}^K \mathcal{H}_k) \leq K - 1 + \sum_{k=1}^K d_{vc}(\mathcal{H}_k)$

3. $\min_{k \in \{1, \ldots, K\}}(d_{vc}(\mathcal{H}_k)) \leq d_{vc}(\bigcup_{k=1}^K \mathcal{H}_k) \leq \sum_{k=1}^K d_{vc}(\mathcal{H}_k)$

4. $\max_{k \in \{1, \ldots, K\}}(d_{vc}(\mathcal{H}_k)) \leq d_{vc}(\bigcup_{k=1}^K \mathcal{H}_k) \leq \sum_{k=1}^K d_{vc}(\mathcal{H}_k)$

5. $\max_{k \in \{1, \ldots, K\}}(d_{vc}(\mathcal{H}_k)) \leq d_{vc}(\bigcup_{k=1}^K \mathcal{H}_k) \leq K - 1 + \sum_{k=1}^K d_{vc}(\mathcal{H}_k)$

\vspace{3mm}

- ¿Cual es la cota más ajustada entre las que son correctas?