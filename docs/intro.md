---
title: Introducción 
subtitle: Curso Reconocimiento de Patrones LCC/UNISON
layout: page
hero_image: https://github.com/ml-unison/ml-unison.github.io/raw/main/docs/img/alt-banner.jpg
hero_darken: true
show_sidebar: false
---


## Presentaciones

1. [¿Qué es el aprendizaje automático?](https://github.com/ml-unison/ml-unison.github.io/raw/main/slides/intro-curso.pptx) 

2. [¿Qué es el aprendizaje supervisado?](https://github.com/ml-unison/ml-unison.github.io/raw/main/slides/intro-supervisado.pdf)

3. [Mi primer modelo de aprendizaje](https://github.com/ml-unison/ml-unison.github.io/raw/main/slides/ejemplo_modelo.pdf) y como saber si mi modelo está bien usando diferentes métricas como [coeficiente de determinación](https://en.wikipedia.org/wiki/Coefficient_of_determination), [matrices de confusión](https://en.wikipedia.org/wiki/Confusion_matrix) o [la curva ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

4. [¿En que sentido el aprendizaje supervisado es posible?](https://github.com/ml-unison/ml-unison.github.io/raw/main/slides/generalizacion.pdf)

5. [El dilema de sesgo y varianza en modelos de aprendizaje (del curso de Caltech)](https://home.work.caltech.edu/slides/slides08.pdf) y una [libretita que lo ilustra](https://github.com/ml-unison/ml-unison.github.io/blob/main/ejemplos/biasvariance.ipynb) 

6. [Presentación sobre generación de características (del curso de IA de Stanford)](https://github.com/IA-UNISON/material/raw/master/presentaciones/non-linear-features.pdf)

7. [Presentación sobre el *overfiting*](http://www.cs.rpi.edu/~magdon/courses/LFD-Slides/SlidesLect11.pdf) de [M. Magdon-Ismail](http://www.cs.rpi.edu/~magdon/)

8. [Presentación sobre regularización](http://www.cs.rpi.edu/~magdon/courses/LFD-Slides/SlidesLect12.pdf) de [M. Magdon-Ismail](http://www.cs.rpi.edu/~magdon/)

9. [Presentación sobre validación](http://www.cs.rpi.edu/~magdon/courses/LFD-Slides/SlidesLect13.pdf) de [M. Magdon-Ismail](http://www.cs.rpi.edu/~magdon/)

10. [Introducción a las MLOps](https://github.com/mcd-unison/aaa-curso/raw/main/slides/MLOps-intro.pdf)


## La librerías para Aprendizaje automático que vamos a usar

1. [*Sci-Kit Learn*](https://scikit-learn.org/), la librería más utilizada (por mucho). Para grandes cantidades de datos [pyspark](https://spark.apache.org/docs/latest/api/python/index.html) incluye el módulo `MLlib` para flujos de aprendizaje automático. Y por último, el nuevo muchacho en el barrio: [pycaret](https://pycaret.org), que ha tenido un crecimiento importante recientemente. 
   

## Lecturas y enlaces recomendados

1. El sitio del curso [*Learning for Data*](https://work.caltech.edu/telecourse.html) de [Yaser S.Abu-Mostafa](https://work.caltech.edu) de CalTech, es probablemente el mejor curso (no edulcorado) que conozco sobre aprendizaje supervisado. Le da un énfasis especial a entender los principios básicos antes de pasar a las técnicas particulares. El sitio incluye
  
     1. Lecturas en *youtube*
  
     2. Las presentaciones en *pdf*
  
     3. Una liga al [libro homónimo del curso](https://www.amazon.com/gp/product/1600490069)
  
     4. Tareas y proyecto
  
2. El [curso de Machine Learning en Coursera](https://www.coursera.org/learn/machine-learning) de [Andrew Ng](https://www.andrewng.org) es posiblemente el curso en linea más popular en Aprendizaje Automático. El profesor es magistral y explica los métodos sin requerir conocimientos en matemáticas, lo que hace que el curso sea a un nivel bastante superficial, pero muy recomendable.

3. [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) el cual se encuentra en la biblioteca de la DCEN y la biblioteca digital. Muy teórico, para quien quiera profundizar mas en las bases del Aprendizaje Automñatico.

4. [An Introduction to Statistical Learning](https://www.statlearning.com) de los mismos autores del libro de *Elements* pero a un nivel mas *undergraduate*, está la versión del libro con ejemplos en R y la versión con ejemplos en python.

4. [Hands-On machine learning with scikit-learn and tensorflow : concepts, tools, and techniques to build intelligent systems](http://www.bibliotecas.uson.mx/sib_acervodet.aspx?bib=15&c1=Titulo&c2=Titulo&t1=Hands-On%20machine%20learning%20with%20scikit-learn%20and%20tensorflow&t2=&cat=) es un texto muy aplicado, con mucho código y poca teoría. Es mi recomendación para buscar como hacer prácticamente cosas y no te gusta buscarlas en internet. Aquí dejo una liga con [libretas jupyter con los ejemplos del libro](https://github.com/ageron/handson-ml3)

5. Los [Tópicos de ML en Medium](https://medium.com/topic/machine-learning) es un buen lugar para buscar artículos de divulgación y/o tutoriales bastante claros sobre diferentes técnicas de ML. Me parece que sin subscripción, tiene uno limitado el número de artículos a leer por día en Medium.
  
## Repositorios de datos con problemas de aprendizaje automático

1. [Kaggle](https://www.kaggle.com) es la puerta de entrada a básicamente la mayor parte de las habilidades en ML que se requieren. El sitio incluye:

    1. Datos reales y sintéticos para practicar y aprender
    
    2. Datos reales en competencias de aprendizaje automático (con premios en efectivo algunas veces)
    
    3. Un sistema de libretas de *jupyter* en linea
    
    4. Cursos sobre diferentes temas interesantes
    
    5. Ejemplos de libretas útiles
    
2. [UCI database repository](https://archive.ics.uci.edu/ml/index.php) Es el sitio más clásico (venerable y feo, por cierto) de conjuntos de datos para tareas de Aprendizaje Automático. Es un poco complicada la navegación y entender la organización de los datos, pero es realmente un muy buen sitio para escoger problemas a resolver.

