---
theme: none
title: Validación
unidad: Unidad 3
layout: portada
subtitulo: Estimar el error fuera de la muestra, y elegir modelos, sin engañarnos
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
  Reconocimiento de Patrones · Unidad 3 · Licenciatura en Ciencias de la Computación, Universidad de Sonora.
---

# Validación

---
layout: pregunta
---

# La cota VC es demasiado holgada y el sesgo y la varianza no se pueden calcular. Entonces, ¿cómo sabemos cuánto vale $E_{\text{out}}$?

<v-click>

Lo **medimos**, con datos que el modelo no vio. La desigualdad de Hoeffding (el promedio de una muestra aleatoria se acerca a la media verdadera) vuelve a funcionar, porque evaluamos una sola hipótesis ya elegida.

</v-click>

---

# El conjunto de validación

<div class="cols" style="gap: 48px">
<div>

Apartamos $M_{\text{val}}$ de los $M$ datos, el conjunto $\Dat_{\text{val}}$. Entrenamos $g^-$ con los $M - M_{\text{val}}$ restantes y medimos

$$
E_{\text{val}}(g^-) = \frac{1}{M_{\text{val}}} \sum_{\vx^{(i)} \in \Dat_{\text{val}}} \ell\big(y^{(i)}, g^-(\vx^{(i)})\big) ,
$$

con $\ell$ la pérdida en un dato.

<v-click>

Como $g^-$ no vio esos $M_{\text{val}}$ datos:

$$
\E\big[E_{\text{val}}(g^-)\big] = \Eout(g^-), \qquad \Var\big[E_{\text{val}}(g^-)\big] = \frac{\sigma^2}{M_{\text{val}}} .
$$

Aquí $\Eout$ es el error fuera de la muestra y $\sigma^2$ la varianza de la pérdida en un dato. Insesgado, y con desviación que baja como $1/\sqrt{M_{\text{val}}}$.

</v-click>
</div>
<div v-click>

### Casas de Ames, error en log del precio

| $M_{\text{val}}$ | desviación de $E_{\text{val}}$ | $\sigma/\sqrt{M_{\text{val}}}$ |
|---|---|---|
| 10 | 0.0116 | 0.0112 |
| 100 | 0.0035 | 0.0035 |
| 1,000 | 0.0011 | 0.0011 |

<div class="pequeno tenue">

Con $\Eout(g^-) = 0.0145$. La teoría y la simulación coinciden.

</div>
</div>
</div>

---

# El dilema de $M_{\text{val}}$

<div class="cols" style="gap: 48px">
<div>

$$
\Eout(g) \;\overset{\text{(pequeño } M_{\text{val}})}{\approx}\; \Eout(g^-) \;\overset{\text{(grande } M_{\text{val}})}{\approx}\; E_{\text{val}}(g^-)
$$

<v-clicks>

- $M_{\text{val}}$ **grande**: $E_{\text{val}}$ es un buen estimador… de un modelo $g^-$ entrenado con pocos datos, peor que $g$ (el modelo entrenado con todos).
- $M_{\text{val}}$ **pequeño**: $g^-$ se parece a $g$, pero $E_{\text{val}}$ es muy ruidoso.
- Regla común: $M_{\text{val}} \approx M/5$. Al final, reentrenar con **todos** los datos.

</v-clicks>
</div>
<div v-click>

### Ames con $M = 600$

| $M_{\text{val}}$ | $\Eout(g^-)$ | desviación de $E_{\text{val}}$ |
|---|---|---|
| 25 | 0.0149 | 0.0064 |
| 200 | 0.0159 | — |
| 400 | 0.0183 | 0.0022 |

<div class="pequeno tenue">

$\Eout(g)$ con los 600 datos: 0.0149.

</div>
</div>
</div>

---
layout: seccion
numero: 1
---

# Elegir contamina

Como la moneda con menos caras entre mil.

---

# Usar la validación para elegir la vuelve optimista

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<Optimismo />

<div v-click="2" style="margin-top: 16px">

La cota de la unión dice cuánto: $\;\Eout \le E_{\text{val}} + O\Big(\sqrt{\tfrac{\ln |\Hyp|}{M_{\text{val}}}}\Big)$, con $|\Hyp|$ el número de modelos comparados. Elegir entre pocos modelos cuesta poco; probar miles, mucho.

</div>

</div>
<div>

Varios modelos que **adivinan al azar** (error real 0.5), evaluados en los mismos 100 datos de validación. Nos quedamos con el de menor error de validación $E_{\text{val}}$.

<div v-click="1">

Con 100 modelos, el elegido parece tener error de 0.37. Es el mismo fenómeno que elegir, entre mil monedas justas, la que sacó menos caras: parece especial sin serlo.

</div>
</div>
</div>

---

# Tres conjuntos, tres trabajos

<div class="cols" style="grid-template-columns: repeat(3, 1fr); gap: 36px">
<div>

### Entrenamiento

Para **ajustar** los parámetros de cada modelo ($\vw$).

</div>
<div v-click>

### Validación

Para **elegir** entre modelos e hiperparámetros ($\lambda$, el grado, la profundidad). Después de usarla, su error ya es optimista.

</div>
<div v-click>

### Prueba

Para **reportar**, una sola vez, al final. Si se usa para decidir algo, deja de ser prueba.

</div>
</div>

<div v-click class="nota">

La versión más común de este error es preprocesar (estandarizar, imputar) con todos los datos antes de partirlos. Elegir el umbral, las variables o el modelo mirando la prueba es la misma trampa.

</div>

---
layout: seccion
numero: 2
---

# Validación cruzada

Que cada dato valide una vez.

---

# Validación cruzada con $K$ particiones

<div class="cols" style="grid-template-columns: 1fr 1fr; gap: 40px">
<div>

<PliegosK />

$$
E_{\text{cv}} = \frac{1}{K} \sum_{k=1}^{K} e_k
$$

donde $e_k$ es el error de validación en la partición $k$.

</div>
<div>

<v-clicks>

- Cada modelo entrena con $M(1 - 1/K)$ datos: casi todos.
- Cada dato valida **una vez**: usamos los $M$ para validar.
- Las $e_k$ no son independientes (comparten datos de entrenamiento), así que $\Var[E_{\text{cv}}]$ no es simplemente $\Var[e_k]/K$. El error estándar que se reporta es aproximado.

</v-clicks>
</div>
</div>

---

# ¿Cuántas particiones?

<div class="cols" style="gap: 48px">
<div>

### Ames, ridge con $\lambda = 10$, 30 semillas

| $K$ | $E_{\text{cv}}$ medio | variación entre semillas |
|---|---|---|
| 2 | 0.01491 | 0.00054 |
| 5 | 0.01429 | 0.00019 |
| 10 | 0.01415 | 0.00014 |
| $M$ (LOO) | 0.01411 | — |

</div>
<div>

<v-clicks>

- Con $K = 2$ cada modelo entrena con la mitad: **sesgo pesimista**.
- Con $K = M$, **dejar uno fuera** (LOO), casi sin sesgo, pero $M$ entrenamientos.
- En la práctica, $K = 5$ o $10$.

</v-clicks>

</div>
</div>

---

# Dejar uno fuera sin reentrenar

<div class="cols" style="gap: 48px">
<div>

Para regresión lineal o ridge, $\hat{\vy} = \mathbf{H} \vy$ con la **matriz sombrero** $\mathbf{H} = \vX(\vX\T \vX + \lambda \mathbf{I})^{-1} \vX\T$, donde $\vX$ es la matriz de datos y $\lambda$ la fuerza de la regularización. Entonces

$$
E_{\text{LOO}} = \frac{1}{M} \sum_{i=1}^{M} \left( \frac{y^{(i)} - \hat y^{(i)}}{1 - H_{ii}} \right)^2 .
$$

<v-click>

Un solo ajuste en lugar de $M$.

</v-click>
</div>
<div v-click>

**Idea de la prueba.** Quitar el punto $i$ cambia $\vX\T \vX$ por $\vX\T \vX - \vx^{(i)} \vx^{(i)\top}$, una corrección de rango 1. Con la fórmula de Sherman–Morrison,

$$
y^{(i)} - g^{(-i)}(\vx^{(i)}) = \frac{y^{(i)} - \hat y^{(i)}}{1 - H_{ii}} ,
$$

con $g^{(-i)}$ el modelo entrenado sin el punto $i$.

<div class="pequeno tenue">

En la libreta: 200 casas, fuerza bruta 0.0132385041 contra fórmula 0.0132385041, 400 veces más rápido. Y la traza $\mathrm{tr}(\mathbf{H})$ son los grados de libertad efectivos del modelo.

</div>
</div>
</div>

---

# Elegir $\lambda$ con validación cruzada

<div class="cols" style="gap: 48px">
<div>

Para cada $\lambda$ de una rejilla, calcular $E_{\text{cv}}(\lambda)$ y su error estándar $\mathrm{EE}$.

<v-clicks>

- **Mínimo:** $\lambda^* = \argmin E_{\text{cv}}(\lambda)$.
- **Regla de una desviación estándar:** el $\lambda$ **más simple** (el mayor) con $E_{\text{cv}}(\lambda) \le E_{\text{cv}}(\lambda^*) + \mathrm{EE}(\lambda^*)$. Si no se distinguen, preferimos el más regularizado: Occam otra vez.

</v-clicks>
</div>
<div v-click>

### Ames, ridge con productos de variables

| regla | $\lambda$ | $E_{\text{cv}}$ |
|---|---|---|
| mínimo | 158 | 0.0150 ± 0.0010 |
| una desviación | 398 | 0.0156 |

<div class="pequeno tenue">

T. Hastie, R. Tibshirani y J. Friedman, *The Elements of Statistical Learning*, sección 7.10.

</div>
</div>
</div>

---

# CV anidada: estimar el error de todo el procedimiento

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<CVAnidada />

</div>
<div>

$E_{\text{cv}}(\lambda^*)$ ya se usó para **elegir**: es optimista. Para saber cuánto vale el procedimiento «elegir $\lambda$ por CV y entrenar», hay que repetir la elección **dentro** de cada partición externa.

<v-click>

| | mínimo | 1 desv. |
|---|---|---|
| $E_{\text{cv}}$ del elegido | 0.0150 | 0.0156 |
| CV anidada | 0.0153 | 0.0159 |
| prueba apartada | 0.0169 | 0.0177 |

</v-click>
</div>
</div>

<!--
Aquí el optimismo es pequeño (0.0003) porque solo se elige un hiperparámetro continuo. Con miles de combinaciones crece. La prueba sale más alta pero dentro de su error estándar (0.003): una sola prueba de 292 casas es insesgada pero ruidosa.
-->

---
layout: seccion
numero: 3
---

# Cuando los datos no son independientes

---

# Facturas de un mismo cliente

<div class="cols" style="grid-template-columns: 3fr 2fr; gap: 40px">
<div>

<PliegosGrupos />

</div>
<div>

Predecir el log del importe de una factura, con un bosque aleatorio. 6,617 facturas de 1,500 clientes.

<v-click>

| variables | KFold | por cliente |
|---|---|---|
| sin id de cliente | 0.315 | 0.325 |
| **con** id de cliente | 0.281 | 0.326 |

</v-click>

<v-click>

La partición aleatoria dice que conocer al cliente ayuda. Por cliente, no ayuda en nada: el modelo **memorizó** clientes que en producción serán nuevos.

</v-click>
</div>
</div>

---

# La partición debe imitar el uso

<div class="cols" style="grid-template-columns: repeat(3, 1fr); gap: 36px">
<div>

### Grupos

Pacientes con varias consultas, estudiantes de la misma escuela, fotos de la misma persona, lecturas del mismo sensor. Si en producción llegan grupos **nuevos**, se valida por grupo (`GroupKFold`).

</div>
<div v-click>

### Tiempo

Si se predice el futuro, se valida con el futuro: entrenar hasta $t$, validar después. Barajar mezcla el pasado con el futuro.

</div>
<div v-click>

### Clases raras

Con 1 % de positivos, una partición puede quedarse sin ninguno. Se estratifica para que cada parte tenga la misma proporción (`StratifiedKFold`).

</div>
</div>

<div v-click class="lema" style="margin-top: 36px; font-size: 27px">
La pregunta no es «¿cómo parto los datos?», sino «¿qué datos <em>verá</em> el modelo cuando lo usemos?».
</div>

---

# Para llevarse

<div class="cols cols-6-4" style="flex: 1">
<div style="font-size: 25px; line-height: 1.4">

1. $E_{\text{val}}$ es insesgado si el modelo no vio esos datos; su ruido baja como $1/\sqrt{M_{\text{val}}}$.
2. **Elegir** con la validación la vuelve optimista: crece con el número de opciones.
3. Validación cruzada: $K = 5$ o $10$ particiones; LOO gratis para modelos lineales.
4. Regla de una desviación estándar y CV anidada.
5. La partición debe imitar cómo se usará el modelo: **grupos**, **tiempo**, **estratos**.

</div>
<div>

### Para practicar

<div class="pequeno">

**Libreta:** *Validación: estimar E_out sin hacer trampa* (Unidad 3, ejemplo 3).

**Laboratorio 3:** validación cruzada y regla de una desviación estándar, desde cero.

**Lectura:** Abu-Mostafa et al., *Learning From Data*, sección 4.3; Hastie et al., *ESL*, sección 7.10.

</div>
</div>
</div>
