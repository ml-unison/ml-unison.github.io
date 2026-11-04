<script setup lang="ts">
// Componentes principales de los 450 dígitos de prueba: en los píxeles y en la capa oculta a[1] de la red 64-64-10.
// Cada clase se marca con su número en la mediana de sus puntos. Datos: herramientas/laminas/u06_01.py.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u06-01.json'
const D = (datos as any).digitos
const P = D.pca
const T = 400, m = 14
function panel(cx: number[], cy: number[]) {
  const ext = (v: number[]) => [Math.min(...v), Math.max(...v)]
  const [x0, x1] = ext(cx), [y0, y1] = ext(cy)
  const x = scaleLinear().domain([x0, x1]).range([m, T - m]), y = scaleLinear().domain([y0, y1]).range([T - m, m])
  const med = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
  const etiquetas = Array.from({ length: 10 }, (_, k) => {
    const idx = P.g.map((g: number, i: number) => g === k ? i : -1).filter((i: number) => i >= 0)
    return { k, x: x(med(idx.map((i: number) => cx[i]))), y: y(med(idx.map((i: number) => cy[i]))) }
  })
  return { puntos: cx.map((v, i) => ({ x: x(v), y: y(cy[i]) })), etiquetas }
}
const paneles = [
  { t: 'píxeles 𝐱', s: D.silueta_x, ...panel(P.x1, P.x2) },
  { t: 'capa oculta 𝐚[1]', s: D.silueta_a, ...panel(P.a1, P.a2) },
]
</script>

<template>
  <div class="ra">
    <div v-for="(p, j) in paneles" :key="j">
      <svg class="grafica" :viewBox="`0 0 ${T} ${T}`" style="background: #F6F7FA; border-radius: 8px">
        <circle v-for="(q, i) in p.puntos" :key="i" :cx="q.x" :cy="q.y" r="3" fill="#8A91A0" opacity="0.45" />
        <text v-for="e in p.etiquetas" :key="e.k" :x="e.x" :y="e.y + 10" text-anchor="middle"
              style="fill: #24398A; font-weight: 800; font-size: 30px; paint-order: stroke; stroke: #F6F7FA; stroke-width: 5px">{{ e.k }}</text>
      </svg>
      <div class="ra__t">{{ p.t }} · silueta {{ p.s.toFixed(2) }}</div>
    </div>
  </div>
</template>

<style scoped>
.ra { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.ra__t { text-align: center; font-weight: 600; font-size: 19px; color: var(--marino); margin-top: 6px; }
</style>
