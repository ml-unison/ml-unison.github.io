---
title: Operaciones de aprendizaje máquina (MLOps)
subtitle: Curso Reconocimiento de Patrones LCC/UNISON
layout: page
hero_image: https://github.com/ml-unison/ml-unison.github.io/raw/main/docs/img/alt-banner.jpg
hero_darken: true
show_sidebar: false
---

## Presentaciones

Empecemos por [esta entrada de blog](https://www.mihaileric.com/posts/mlops-is-a-mess/) que posiciona las *MLOps* de forma sencilla y amena.


### Presentaciones del curso Machine Learning Systems Design de Stanford

Las presentaciones son tomadas del curso [CS 329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/syllabus.html) de Stanford, impartida por [Chip Huyen](https://huyenchip.com). La selección de las presentaciones se hizo originalmente para la *Maestría en Ciencia de Datos* pero mme parece que toda la información es relevante para la *Licenciatura en Ciencias de la Computación*.


1. [Introducción a las MLOps](https://github.com/mcd-unison/aaa-curso/raw/main/slides/MLOps-intro.pdf)
   
2. [Preprocesamiento de datos](https://github.com/mcd-unison/aaa-curso/raw/main/slides/mlops-preprocesamiento.pdf)

3. [Selección de modelos de ML](https://github.com/mcd-unison/aaa-curso/raw/main/slides/mlops-seleccion.pdf)

4. [Despliegue de modelos de ML](https://github.com/mcd-unison/aaa-curso/raw/main/slides/mlops-deploy.pptx)

5. [Monitoreo de la calidad de los modelo](https://github.com/mcd-unison/aaa-curso/raw/main/slides/mlops-monitor.pptx)

6. [Una presentación de *Weights and Biases*](https://github.com/mcd-unison/aaa-curso/raw/main/slides/WandB-tutorial.pptx)

7. [Infraestructura y plataformas](https://github.com/mcd-unison/aaa-curso/raw/main/slides/MLOps-infraestructura.pdf)


## Libretas, scripts y tutoriales

1. [Un ejemplo en github](https://github.com/juliowaissman/github-mlflow-dagshub-colab) para usar [DagsHub](https://dagshub.com/) para manejar proyectos colaborativos con *MLFlow* (El repositorio también se puede revisar en su [sincronización en DagsHub](https://dagshub.com/juliowaissman/github-mlflow-dagshub-colab) y su [*MLFlow Tracking Dashboard*](https://dagshub.com/juliowaissman/github-mlflow-dagshub-colab.mlflow)).

2. Enlace a la página de [OnDemand](https://ondemand-acarus.unison.mx) del ACARUS (requieren de su cuenta y su contraseña). En *OnDemand* se tiene instalado MLOps para servir un *Tracking Server* (y se puede ver usando [esta libreta de ejemplo](https://github.com/mcd-unison/aaa-curso/raw/main/ejemplos/autologgers.ipynb))

3. [Made with ML](https://madewithml.com). Un curso muy interesante con una prespectiva industrial del aprendizaje automático. Mucho le dedica a las MLOps.

4. [Ejemplo de uso de *Ray Serve* para desplegar modelos](https://github.com/anyscale/academy/blob/main/ray-serve/e2e/tutorial.ipynb)

5. [Ejemplo de uso de *Evidently* para monitorear los datos](https://github.com/anyscale/academy/blob/main/ray-serve/e2e/tutorial.ipynb)


## Herramientas para MLOps

1. [MLFlow](https://www.mlflow.org). La plataforma de código abierto más usada como base para esquemas de MLOps. Una versión simple de lo que puedes tener con [*DataBricks*](https://databricks.com) si estás en una compañía que esté dispuesta a pagar el precio. Existe una versión para experimentación.

2. [*Weights & Biases*](https://wandb.ai/site). Similar a DataBricks, la solucion completa. Existe una versión académica y una personal, pero no puede usarse en empresas.

3. [*MLRun*](https://www.mlrun.org). Otra plataforma similar a *MLFlow* pero incluye una interfase gráfica para el manejo del ciclo de vida (como *DataBricks*). Puede instalarse *on premise*.
   
4. [*Neptune.ai*](https://neptune.ai). Otra plataforma más, en la nube la orquestación, pero los modelos se pueden ejecutar *on premise*.

5. [*Ray*](https://www.ray.io/). Una plataforma de desarrollo pensando en la escalabilidad sobre todo. EN particular es muy apreciada la librería *ray serve* para poner los modelos de aprendizaje en producción.

6. [*MetaFlow*](https://metaflow.org). La solución de *Netflix* para MLOps. Al parecer muy poderoso pero tambien muy complicado de instalar *on premise*. Funciona sobre *Kubernetes*.

7. [*Facets*](https://pair-code.github.io/facets/). Otra herramienta de EDA, muy importante saber las estadísticas básicas de los datos para poder monitorearlos con [*pydantic*](https://pydantic-docs.helpmanual.io) o [*Great Expectations*](https://github.com/great-expectations/great_expectations).

8. [*Great Expectations*](https://github.com/great-expectations/great_expectations) y [*pydantic*](https://pydantic-docs.helpmanual.io). Validación de datos conforme los utiliza uno, exclusivo par *python*. 

9.  [*Evidently AI*](https://evidentlyai.com). Herramientas en código abierto para monitorear la calidad de los datos, la deriva de los datos y la calidad de los modelos de predicción durante la etapa de producción. Se puede integrar a MLFlow, pero están más bonitos los tableros de *Evidently*.

10. [*Whylogs*](https://github.com/whylabs/whylogs). Similar a *Evidently* al parecer, pero solo la librería, ya que es el corazón de un producto que ya no es de código abierto (*Whylabs*). 

11. [*AirFlow*](https://airflow.apache.org). No es propiamente para MLOps, pero se utiliza mucho en conjunto con otras plataformas. Para ejecutar *workflows* con ETLs, para automatizar los procesos de predicción y de reentrenamiento en sistemas de aprendizaje automñatico.
