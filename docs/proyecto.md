---
title: Proyecto integrador
subtitle: Curso Reconocimiento de Patrones LCC/UNISON
layout: page
hero_image: https://github.com/ml-unison/ml-unison.github.io/raw/main/docs/img/alt-banner.jpg
hero_darken: true
show_sidebar: false
---


Este proyecto integrador de basa en dos competencias de Kaggle, en la que deberán competir en equipos de mínimo 1 y máximo 3 integrantes, y entregar lo que se describe en la sección de *Entregables*. 

## Predict the Introverts from the Extroverts

La competencia de puede acceder [**en este enlace**](https://www.kaggle.com/competitions/playground-series-s5e7) 

¡Bienvenidos a la Serie de Desafíos de Kaggle 2025! Planeamos continuar con el espíritu de las ediciones anteriores, ofreciendo conjuntos de datos interesantes y accesibles para que nuestra comunidad practique sus habilidades de aprendizaje automático, y anticipamos una competencia cada mes.

Tu objetivo: Predecir si una persona es introvertida o extrovertida, considerando su comportamiento social y rasgos de personalidad.

El objetivo de la serie *Tabular Playground* es ofrecer a la comunidad de Kaggle una variedad de desafíos relativamente sencillos que permitan aprender y perfeccionar habilidades en diferentes aspectos del aprendizaje automático y la ciencia de datos. Estos desafíos utilizan conjuntos de datos relativamente sencillos, generados sintéticamente a partir de datos reales, y brindan la oportunidad de experimentar rápidamente con diversas ideas de modelado y diseño de características, crear visualizaciones, etc.

El uso de datos sintéticos para las competiciones de Playground nos permite encontrar un equilibrio entre disponer de datos del mundo real (con características identificadas) y garantizar que las etiquetas de prueba no sean de acceso público. Esto nos permite organizar competiciones con conjuntos de datos más interesantes que antes. 



## CSIRO - Image2Biomass Prediction

La competencia de puede acceder [**en este enlace**](https://www.kaggle.com/competitions/csiro-biomass) 

Los ganaderos suelen entrar en un potrero y hacerse una pregunta: "¿Hay suficiente pasto para el ganado?". Parece sencillo, pero la respuesta dista mucho de serlo. La biomasa del pasto —la cantidad de alimento disponible— determina cuándo pueden pastar los animales, cuándo los campos necesitan un descanso y cómo mantener la productividad de los pastos temporada tras temporada.

Si la estimación es incorrecta, la tierra sufre; el alimento se desperdicia y los animales tienen dificultades. Si es correcta, todos ganan: mayor bienestar animal, producción más constante y suelos más sanos.

Los métodos actuales dificultan esta evaluación innecesariamente. El método tradicional de "cortar y pesar" es preciso, pero lento e inviable a gran escala. Los medidores de placa y los capacitivos ofrecen lecturas más rápidas, pero no son fiables en condiciones variables. La teledetección permite una monitorización a gran escala, pero aún requiere validación manual y no puede diferenciar la biomasa por especies.

Este concurso te reta a aportar soluciones más sostenibles al campo: crea un modelo que prediga la biomasa del pasto a partir de imágenes, mediciones de campo y conjuntos de datos disponibles públicamente. Trabajarás con un conjunto de datos anotados profesionalmente que abarca los pastos australianos en diferentes estaciones, regiones y combinaciones de especies, junto con valores NDVI para mejorar tus modelos.

Si tienes éxito, no solo mejorarás los métodos de estimación, sino que también ayudarás a los ganaderos a tomar decisiones de pastoreo más acertadas, permitirás a los investigadores realizar un seguimiento más preciso de la salud de los pastos e impulsarás al sector agrícola hacia sistemas más sostenibles y productivos.

## Entregables (por competencia)

1. Cada equipo deberá estar registrado en *Kaggle* con las cuentas de cada uno de ustedes. 
2. Ya sea en un repositorio de *GitHub* o dentro del mismo espacio de *Kaggle* (asociado con la competencia), subir una o varias libretas *públicas* con todo el proceso que siguieron. Lo que deberá tener como mínimo la, o las libretas es lo siguiente:
   1. Análisis exploratorio sencillo de los datos (que datos tenemos, que tipo son los datos, si hay o no datos faltantes, si hay datos numéricos, que distribución tienen, y si hay cualitativos si son ordenados o no por ejemplo).
   2. Algún método de reagrupamiento, o visualización con PCA o t-SNE de los datos de entrenamiento, que permita ver si hay ciertos patrones claros.
   3. Preprocesamiento de los datos, y su codificación para usarlos como entrada a un modelo. Hay que tener en cuenta que este preprocesamiento debe poder integrarse mediante *pipelines* al modelo final.
   4. Decidir si es un problema de clasificación o de regresión (ambos podrían aplicar) y justificar porqué.
   5. Utilizar un modelo inicial simple que permita establecer un *baseline* del problema.
   6. Usar algún método de búsqueda de hiperparámetros y seleccionar el modelo de aprendizaje que consideres más adecuado. Justificar la selección.
   7. Revisar y mostrar la calidad del modelo retenido usando métodos (los que consideren mejor adaptados) como *k-cross-fold-validation*, curvas de aprendizaje, curvas RoC, matrices de confusión o métricas de entrenamiento y validación.
3. Anexar el enlace del *leaderboard* con su evaluación final como equipo. Recuerden que pueden someter varias modelos, pero al final tendrán que escoger uno solo para su evaluación final.
