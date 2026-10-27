// Nombres legibles de los atributos de Ames (u05-04) y escala de color divergente bajo → alto.
export const NOMBRES: Record<string, string> = {
  area: 'área habitable', calidad: 'calidad general', anio: 'año de construcción', remodelacion: 'año de remodelación',
  sotano: 'sótano', cochera: 'cochera', terreno: 'terreno', banos: 'baños', chimeneas: 'chimeneas', cocina: 'cocina',
  condicion: 'condición', barrio: 'barrio', area_vendedor: 'área (vendedor)',
}
export const UNIDADES: Record<string, string> = { area: ' m²', sotano: ' m²', terreno: ' m²', cochera: ' autos', barrio: ' mil USD' }

const hex = (c: string) => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16))
const A = hex('#24398A'), B = hex('#E4E6EC'), R = hex('#D5433C')
/** t en [0, 1]: azul (bajo) → gris claro → rojo (alto). */
export function divergente(t: number): string {
  t = Math.min(1, Math.max(0, t))
  const [p, q, s] = t < 0.5 ? [A, B, 2 * t] : [B, R, 2 * t - 1]
  return `rgb(${p.map((v, i) => Math.round(v + (q[i] - v) * s)).join(',')})`
}
