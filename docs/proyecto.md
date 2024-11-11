---
title: Proyecto integrador
subtitle: Curso Reconocimiento de Patrones LCC/UNISON
layout: page
hero_image: https://github.com/ml-unison/ml-unison.github.io/raw/main/docs/img/alt-banner.jpg
hero_darken: true
show_sidebar: false
---

## Child Mind Institute — Problematic Internet Use

Este proyecto integrador de basa en una competencia de Kaggle, en la que deberán competir en equipos de mínimo 1 y máximo 3 integrantes, y entregar lo que se describe en la sección de *Entregables*. La competencia de puede acceder [**en este enlace**](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) 

En la era digital actual, el uso problemático de internet entre niños y adolescentes es una preocupación creciente. Comprender mejor este problema es crucial para abordar problemas de salud mental como la depresión y la ansiedad.

Los métodos actuales para medir el uso problemático de internet en niños y adolescentes suelen ser complejos y requieren evaluaciones profesionales. Esto crea barreras de acceso, culturales y lingüísticas para muchas familias. Debido a estas limitaciones, el uso problemático de internet a menudo no se mide directamente, sino que se asocia con problemas como la depresión y la ansiedad en los jóvenes.

Por el contrario, las mediciones de condición física son extremadamente accesibles y están ampliamente disponibles con una intervención mínima o sin necesidad de experiencia clínica. Los cambios en los hábitos físicos, como una peor postura, una dieta irregular y una actividad física reducida, son comunes en los usuarios excesivos de tecnología. Proponemos usar estos indicadores de condición física, que son fáciles de obtener, como proxies para identificar el uso problemático de internet, especialmente en contextos donde falta experiencia clínica o herramientas de evaluación adecuadas.

Este concurso te desafía a desarrollar un modelo predictivo capaz de analizar datos de actividad física de los niños para detectar indicadores tempranos de uso problemático de internet y tecnología. Esto permitirá intervenciones tempranas destinadas a promover hábitos digitales más saludables.


## Entregables

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
