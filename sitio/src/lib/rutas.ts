import { curso } from '../datos/curso'

const BASE = import.meta.env.BASE_URL.replace(/\/?$/, '/')

/** Ruta interna con el prefijo del sitio ("/" o "/preview/"). */
export const url = (ruta = '') => BASE + ruta.replace(/^\//, '')

/** Enlace para abrir una libreta del repositorio en Google Colab. */
export const colab = (ruta: string) =>
  `https://colab.research.google.com/github/${curso.repo}/blob/${curso.rama}/${ruta}`

/** Presentación (Slidev) publicada junto al sitio. */
export const presentacion = (id: string) => url(`presentaciones/${id}/`)
