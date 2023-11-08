---
title: Estrés hídrico en el estado de Sonora
subtitle: Curso Reconocimiento de Patrones LCC/UNISON
layout: page
hero_image: https://github.com/ml-unison/ml-unison.github.io/raw/main/docs/img/alt-banner.jpg
hero_darken: true
show_sidebar: false
---

## Objetivo

Este miniproyecto tiene como objetivo enfrentarlos a una problemática real, con un usuario real, y con datos reales. En este proyecto se utilizarán los datos que les ofrecemos pero también se pueden consultar otras fuentes de datos abiertos.

Agradecemos a la Directora de Inteligencia de Datos del Gobierno del Estado de Sonora, Diana Ballesteros, por el apoyo y seguimientos para este proyecto, así como su ayuda en la evaluación de las propuestas que entreguen los diferentes equipos, como usuaria final del trabajo.

----

## Problemática

En Sonora hay un problema real sobre la disponibilidad de recursos hídricos para el consumo humano, la industria, la agricultura y la ganadería. Para que el gobierno pueda evaluar e implementar políticas públicas adecuadas, es necesario poder predecir con márgenes de error bien establecidos, cuales van a ser los recursos hídricos que se dispongan en un futuro.

Entre las fuentes de información con la que se cuenta, podemos enumerar las siguientes:

- [Catálogo estatal de presas](https://github.com/ml-unison/ml-unison.github.io/raw/main/proyecto/catalogo_estatal.xlsx)
- [Almacenamiento histórico en presas de Sonora](https://github.com/ml-unison/ml-unison.github.io/raw/main/proyecto/almac_reciente.xlsx)

Una exploración rápida de estos archivos la pueden visualizar en [esta libreta de jupyter](https://github.com/ml-unison/ml-unison.github.io/raw/main/proyecto/exploracion.ipynb), realizada por su compañero Jesúsu Enrique Escobedo del Castillo.

También puede buscarse datos meteorológicos en la CONAGUA, la NASA o el NOA (que si bien son instituciones de los EE.UU., tienen un seguimiento que alcanza al estado de Sonora)

----

## Proyecto

Para este proyecto se espera que cada equipo haga lo siguiente:

1. Definir, a partir de la información que se tiene y de la que encuentren en fuentes públicas, cual es la problemática que proponen atacar.
2. Definir que tipo de problema de ML es (clasificación, regresión, regrupamiento, ...) y que se establezacan métricas para medir la calidad de los modelos.
3. Preprocesamiento de los datos, decidir si se hacen series de prueva y validación o si se utiliza una estrategia de *K-fold-cross-validation*
4. Ajuste de modelos, utilizando *mlflow* para tener un seguimiento y un registro de todas las pruebas realizadas.
5. Justificación de la desición del modelo seleccionado en términos de aprendizaje automñatico (¿Generaliza el modelo? ¿Particulariza?)
6. Reporte final con resultados en formato ejecutivo.

El reporte final deberá tener los siguiente componentes:

1. Título y nombre de los autores con su contacto (correo electrónico)
2. Resumen. Un sólo párrafo que muestre los resultados principales y las conclusiones que se llegaron. El resumen se debe acompañar con una sola figura con los resultados más importantes a mostrar (una sola).
3. Detalles técnicos. Una explicación más detallada de lo que se hizo. 
 
Con todo y figuras, tablas y referencias, el documento final no puede exceder 3 páginas, usando letra 12, o 11 como muy pequeña, y márgenes decentes de la hoja.

----




