<script setup lang="ts">
// Esquema (no los datos) del árbol de Cann, Stoneking y Wilson (1987): la raíz se abre en dos
// ramas; una solo tiene tipos africanos y la otra mezcla las cinco poblaciones.
import { SERIE } from './colores'
const regiones = ['África', 'Asia', 'Europa', 'Australia', 'Nueva Guinea']
const col = [SERIE[0], SERIE[1], SERIE[2], SERIE[3], '#8DA3C9']
// hojas de arriba abajo: región de cada una
const hojas = [0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 3, 3, 0, 2, 2, 1, 4, 4, 1, 0, 2, 4, 3, 1, 2, 0]
// subárboles: [primera hoja, última hoja, profundidad a la que se unen]  (0 = raíz, 1 = hojas)
const nodos: [number, number, number][] = [
  [0, 5, 0.25], [0, 2, 0.55], [3, 5, 0.6],
  [6, 25, 0.22], [6, 11, 0.42], [6, 8, 0.7], [9, 11, 0.62], [12, 18, 0.45], [12, 14, 0.72], [15, 18, 0.6], [19, 25, 0.38], [19, 21, 0.66], [22, 25, 0.58],
]
const W = 700, H = 500, x0 = 30, x1 = 480, top = 20, paso = 16.8
const X = (p: number) => x0 + p * (x1 - x0)
const Y = (i: number) => top + i * paso
// cada subárbol: vertical en su profundidad, que abarca a sus hijos inmediatos, y horizontales hacia ellos
function hijos(n: [number, number, number]) {
  const dentro = nodos.filter(m => m !== n && m[0] >= n[0] && m[1] <= n[1] && !(m[0] === n[0] && m[1] === n[1]))
  const directos = dentro.filter(m => !dentro.some(o => o !== m && o[0] <= m[0] && o[1] >= m[1] && !(o[0] === m[0] && o[1] === m[1])))
  const cubiertas = new Set(directos.flatMap(m => Array.from({ length: m[1] - m[0] + 1 }, (_, k) => m[0] + k)))
  const sueltas = Array.from({ length: n[1] - n[0] + 1 }, (_, k) => n[0] + k).filter(i => !cubiertas.has(i))
  return [...directos.map(m => ({ y: centro(m), x: X(m[2]) })), ...sueltas.map(i => ({ y: Y(i), x: X(1) }))]
}
function centro(n: [number, number, number]): number { const h = hijos(n); return (Math.min(...h.map(c => c.y)) + Math.max(...h.map(c => c.y))) / 2 }
const lineas = nodos.flatMap(n => { const h = hijos(n); const ys = h.map(c => c.y)
  return [{ x1: X(n[2]), x2: X(n[2]), y1: Math.min(...ys), y2: Math.max(...ys) }, ...h.map(c => ({ x1: X(n[2]), x2: c.x, y1: c.y, y2: c.y }))] })
const raiz = { y: (centro(nodos[0]) + centro(nodos[3])) / 2 }
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <line :x1="X(0)" :x2="X(0)" :y1="centro(nodos[0])" :y2="centro(nodos[3])" stroke="#20305A" stroke-width="2.4" />
    <line :x1="X(0)" :x2="X(0.25)" :y1="centro(nodos[0])" :y2="centro(nodos[0])" stroke="#20305A" stroke-width="2.4" />
    <line :x1="X(0)" :x2="X(0.22)" :y1="centro(nodos[3])" :y2="centro(nodos[3])" stroke="#20305A" stroke-width="2.4" />
    <line v-for="(l, i) in lineas" :key="i" v-bind="l" stroke="#20305A" stroke-width="2" />
    <circle :cx="X(0)" :cy="raiz.y" r="9" fill="#C9971C" stroke="#fff" stroke-width="2" />
    <text :x="X(0) + 14" :y="raiz.y - 2" style="fill: #1A2133; font-weight: 600; font-size: 16px">ancestra</text>
    <text :x="X(0) + 14" :y="raiz.y + 17" style="fill: #1A2133; font-weight: 600; font-size: 16px">común</text>
    <circle v-for="(r, i) in hojas" :key="`h${i}`" :cx="X(1) + 10" :cy="Y(i)" r="6.5" :fill="col[r]" />
    <text :x="X(1) + 26" :y="(Y(0) + Y(5)) / 2 + 6" style="fill: #1A2133; font-weight: 600">solo africanos</text>
    <text :x="X(1) + 26" :y="(Y(6) + Y(25)) / 2 + 6" style="fill: #1A2133; font-weight: 600">las cinco poblaciones</text>
    <g v-for="(n, i) in regiones" :key="n" :transform="`translate(${10 + i * 132}, ${H - 8})`">
      <circle r="6.5" cy="-6" :fill="col[i]" /><text x="12" y="0" style="font-size: 16px; fill: #1A2133">{{ n }}</text>
    </g>
  </svg>
</template>
