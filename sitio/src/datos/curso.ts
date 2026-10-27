// Datos generales del curso. Todo lo que cambia cada semestre está aquí.
export const curso = {
  nombre: 'Reconocimiento de Patrones',
  programa: 'Licenciatura en Ciencias de la Computación',
  departamento: 'Departamento de Matemáticas',
  universidad: 'Universidad de Sonora',
  semestre: '2027-1',
  profesor: 'Julio Waissman',
  correo: 'julio.waissman@unison.mx',
  horario: 'Lunes a viernes, 13:00–14:00',
  lugar: 'Salón 101, edificio 3K-4',
  repo: 'ml-unison/ml-unison.github.io',
  // rama de la que Colab toma las libretas; la vista previa usa la rama de desarrollo
  rama: process.env.RDP_RAMA ?? 'main',
}

export interface Unidad {
  num: number
  slug: string
  titulo: string
  semanas: string
  resumen: string
  temas: string[]
  disponible: boolean
}

export const unidades: Unidad[] = [
  {
    num: 1, slug: '01-introduccion', titulo: 'Introducción y flujo de trabajo', semanas: '2',
    resumen: 'Qué significa aprender de datos, el ciclo de vida de un modelo y cómo evaluar sin engañarnos.',
    temas: ['Paradigmas de aprendizaje', 'Riesgo empírico y ERM', 'Fuga de información', 'Ética y sesgo', 'pandas', 'Ingeniería de atributos y pipelines'],
    disponible: true,
  },
  {
    num: 2, slug: '02-no-supervisado', titulo: 'Aprendizaje no supervisado', semanas: '2',
    resumen: 'Encontrar estructura sin etiquetas: agrupamiento, mezclas y reducción de dimensión.',
    temas: ['k-medias', 'Agrupamiento jerárquico', 'DBSCAN', 'Mezclas gaussianas y EM', 'PCA', 't-SNE y UMAP'],
    disponible: false,
  },
  {
    num: 3, slug: '03-fundamentos', titulo: 'Fundamentos del aprendizaje supervisado', semanas: '2.5',
    resumen: 'Por qué es posible generalizar y cómo medirlo con honestidad.',
    temas: ['Hoeffding y dimensión VC', 'Sesgo y varianza', 'Regularización', 'Validación cruzada', 'Métricas, ROC y calibración'],
    disponible: false,
  },
  {
    num: 4, slug: '04-lineales', titulo: 'Modelos lineales', semanas: '2',
    resumen: 'La familia de modelos más útil y mejor entendida, desde mínimos cuadrados hasta los kernels.',
    temas: ['Regresión lineal', 'Regresión logística', 'GLM', 'LDA y Bayes ingenuo', 'SVM y kernels'],
    disponible: false,
  },
  {
    num: 5, slug: '05-arboles-ensambles', titulo: 'Árboles y ensambles', semanas: '2',
    resumen: 'Los modelos que dominan los datos tabulares, y cómo explicar lo que hacen.',
    temas: ['Árboles de decisión', 'Bagging y bosques aleatorios', 'Boosting', 'XGBoost y LightGBM', 'SHAP e interpretabilidad'],
    disponible: false,
  },
  {
    num: 6, slug: '06-redes-neuronales', titulo: 'Redes neuronales', semanas: '3',
    resumen: 'Del perceptrón multicapa a la atención, con PyTorch.',
    temas: ['MLP y backprop', 'Optimización', 'Redes convolucionales', 'Transferencia', 'RNN, atención y transformers'],
    disponible: false,
  },
  {
    num: 7, slug: '07-series-tiempo', titulo: 'Series de tiempo', semanas: '1.5',
    resumen: 'Pronosticar respetando el orden del tiempo.',
    temas: ['Validación temporal', 'AR y ARIMA', 'Atributos rezagados con boosting', 'Modelos neuronales'],
    disponible: false,
  },
  {
    num: 8, slug: '08-mlops', titulo: 'MLOps y despliegue', semanas: '1',
    resumen: 'Llevar un modelo a producción y mantenerlo vivo.',
    temas: ['MLflow', 'Registro de modelos', 'Servicio con FastAPI', 'Monitoreo y drift'],
    disponible: false,
  },
]

export const dosDigitos = (n: number) => String(n).padStart(2, '0')
