---
title: Operaciones de aprendizaje máquina (MLOps)
subtitle: Curso Reconocimiento de Patrones LCC/UNISON
layout: page
hero_image: https://github.com/ml-unison/ml-unison.github.io/raw/main/docs/img/alt-banner.jpg
hero_darken: true
show_sidebar: false
---

## Presentaciones

1. [Una presentación inicial](https://github.com/mcd-unison/aaa-curso/raw/main/slides/mlops_intro.pdf)
2. [Una presentación breve de MLFlow y su uso básico](https://github.com/mcd-unison/aaa-curso/raw/main/slides/mlflow_recortes.pdf)


## Herramientas básicos

1. [MLFlow](https://www.mlflow.org/). La plataforma de código abierto más usada como base para esquemas de MLOps.
2. [DVC](https://dvc.org). Originalmente para versionado de datos, aunque ya hacen un poco de todo. Es mejor combinarlo con *MLFlow*, si no se quiere pasar a un contenedor con [DeltaLake](https://www.datacamp.com/tutorial/delta-lake?utm_cid=21057859163&utm_aid=157296750617&utm_campaign=230119_1-ps-other~dsa~tofu_2-b2c_3-latam-en_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9133003-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~latam-en~dsa~tofu~tutorial~data-engineering&gad_source=1&gad_campaignid=21057859163&gbraid=0AAAAADQ9WsEZedSIX5pYRZTooJ8wKf49V&gclid=Cj0KCQiAxJXJBhD_ARIsAH_JGjhiDHcgKSiSDj_Nemc5QM9rGK5uy8B6avyfb9EzQx9BTZwlaxyAbWQaAo4UEALw_wcB).
3. [RAY](https://www.ray.io). Un competidor de *MLFlow*, que muchas veces se usan en conjunto. Versión en nube con *Anyscale* o versión libre para ejecutarse en local.
4. [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/). Un esquema lógico y sencillo para plantear proyectos de ciencia de datos en python en forma genérica.