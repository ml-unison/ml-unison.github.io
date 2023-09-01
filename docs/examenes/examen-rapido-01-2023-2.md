---
title: Examen rápido No. 1
author: Reconocimiento de Patrones (2023-2)
date: Julio Waissman Vilanova
---

\vspace{3mm}

**Nombre:** _______________________________________________________

# Problema 1

La desigualdad de Hoeffding es un medio para caracterizar el error fuera de muestra de una hipótesis de aprendizaje a partir de una cota superior:

$$
P\left[ |E_{out}(h^*) - E_{in}(h^*)| > \varepsilon \right] \leq 2 |\mathcal{H}| e^{-2 \varepsilon^2 M}, \quad \varepsilon > 0.
$$

Si establecemos que $\varepsilon = 0.05$ y queremos que la cota superior $2 |\mathcal{H}| e^{-2 \varepsilon^2 M}$ sea a lo más $0.03$, ¿Cuál sería el número mínimo de datos de entrenamiento $M$ de pendiendo de la cardinalidad del modelo de aprendizaje?

\vspace{3mm}

- Para el caso $|\mathcal{H}| = 1$: 

\vspace{3mm}

- Para el caso $|\mathcal{H}| = 1$: 

\vspace{3mm}

- Para el caso $|\mathcal{H}| = 1$: 

\newpage

# Problema 2

Consideremos el modelo de aprendizaje que vamos a llamar *2--intervalos*. En este modelo $\mathcal{H}$ vamos a considerar que:

$$
h: \mathrm{R} \to \{0, 1\},
$$
donde $h(x) = +1$ si el punto $x \in \mathrm{R}$ se encuentra dentro de alguno de dos intervalos (i.e. $[a, b]$ y $[c, d]$) preestablecidos. En caso contrario, $h(x) = -1$.

\vspace{3mm}

- ¿Cual es el breakpoint de $\mathcal{H}$?

\vspace{3mm}

- ¿Que valor tiene $d_{VC}(\mathcal{H})$?

\vspace{3mm}

Ahora consideremos el caso genérico *$M$--intervalos*, donde $\mathcal{H}$ está definido como el cinjunto de funciones $h: \mathrm{R} \to \{0, 1\}$ tales que $h(x) = +1$ si $x$ se encuentra en alguno de los $M$ intervalos establecidos y $h(x) = -1$ en caso contrario.

\vspace{3mm}

- ¿Que valor tiene $d_{VC}(\mathcal{H})$?


