// Cursos y proyectos de DataCamp que forman parte de la evaluación, por unidad.
// Las ligas se verificaron en octubre de 2026.
export type TipoDataCamp = 'Curso' | 'Proyecto' | 'Ruta' | 'Evaluación'
export interface ActividadDataCamp { titulo: string; tipo: TipoDataCamp; horas?: number; url?: string }

const dc = 'https://www.datacamp.com'

export const datacamp: Record<string, ActividadDataCamp[]> = {
  '01-introduccion': [
    { titulo: 'Understanding Machine Learning', tipo: 'Curso', horas: 2, url: `${dc}/courses/understanding-machine-learning` },
    { titulo: 'Preprocessing for Machine Learning in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/preprocessing-for-machine-learning-in-python` },
    { titulo: 'Feature Engineering for Machine Learning in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/feature-engineering-for-machine-learning-in-python` },
    { titulo: 'Powering Data for the Department of Energy: Building an ETL Pipeline', tipo: 'Proyecto', horas: 1, url: `${dc}/projects/2706` },
  ],
  '02-no-supervisado': [
    { titulo: 'Unsupervised Learning in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/unsupervised-learning-in-python` },
    { titulo: 'Cluster Analysis in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/cluster-analysis-in-python` },
    { titulo: 'Dimensionality Reduction in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/dimensionality-reduction-in-python` },
    { titulo: 'Clustering Antarctic Penguin Species', tipo: 'Proyecto', horas: 1, url: `${dc}/projects/1809` },
  ],
  '03-fundamentos': [
    { titulo: 'Supervised Learning with scikit-learn', tipo: 'Curso', horas: 4, url: `${dc}/courses/supervised-learning-with-scikit-learn` },
    { titulo: 'Model Validation in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/model-validation-in-python` },
    { titulo: 'Hyperparameter Tuning in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/hyperparameter-tuning-in-python` },
    { titulo: 'Winning a Kaggle Competition in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/winning-a-kaggle-competition-in-python` },
    { titulo: 'Machine Learning Fundamentals in Python', tipo: 'Evaluación' },
  ],
  '04-lineales': [
    { titulo: 'Linear Classifiers in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/linear-classifiers-in-python` },
    { titulo: 'Predictive Modeling for Agriculture', tipo: 'Proyecto', horas: 1, url: `${dc}/projects/1772` },
  ],
  '05-arboles-ensambles': [
    { titulo: 'Machine Learning with Tree-Based Models in Python', tipo: 'Curso', horas: 5, url: `${dc}/courses/machine-learning-with-tree-based-models-in-python` },
    { titulo: 'Extreme Gradient Boosting with XGBoost', tipo: 'Curso', horas: 4, url: `${dc}/courses/extreme-gradient-boosting-with-xgboost` },
    { titulo: 'Predicting Movie Rental Durations', tipo: 'Proyecto', horas: 1, url: `${dc}/projects/1796` },
  ],
  '06-redes-neuronales': [
    { titulo: 'Introduction to Deep Learning with PyTorch', tipo: 'Curso', horas: 4, url: `${dc}/courses/introduction-to-deep-learning-with-pytorch` },
    { titulo: 'Intermediate Deep Learning with PyTorch', tipo: 'Curso', horas: 4, url: `${dc}/courses/intermediate-deep-learning-with-pytorch` },
  ],
  '07-series-tiempo': [
    { titulo: 'Machine Learning for Time Series Data in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/machine-learning-for-time-series-data-in-python` },
  ],
  '08-mlops': [
    { titulo: 'Machine Learning in Production', tipo: 'Ruta', horas: 19, url: `${dc}/tracks/machine-learning-in-production` },
  ],
}

// Puntos extra: la ruta profesional completa más estos cursos.
export const rutaCompleta: ActividadDataCamp = {
  titulo: 'Machine Learning Scientist in Python', tipo: 'Ruta', horas: 85, url: `${dc}/tracks/machine-learning-scientist-with-python`,
}
export const extra: ActividadDataCamp[] = [
  { titulo: 'Natural Language Processing (NLP) in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/natural-language-processing-nlp-in-python` },
  { titulo: 'Natural Language Processing with spaCy', tipo: 'Curso', horas: 4, url: `${dc}/courses/natural-language-processing-with-spacy` },
  { titulo: 'Feature Engineering for NLP in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/feature-engineering-for-nlp-in-python` },
  { titulo: 'Image Processing in Python', tipo: 'Curso', horas: 4, url: `${dc}/courses/image-processing-in-python` },
  { titulo: 'Introduction to PySpark', tipo: 'Curso', horas: 4, url: `${dc}/courses/introduction-to-pyspark` },
  { titulo: 'Machine Learning with PySpark', tipo: 'Curso', horas: 4, url: `${dc}/courses/machine-learning-with-pyspark` },
]
