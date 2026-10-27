<script setup lang="ts">
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import { AZUL, DORADO, ROJO } from './colores'

const W = 560, H = 380, m = { l: 60, r: 20, t: 36, b: 70 }
const x = scaleLinear().domain([0, 800]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 1]).range([H - m.b, m.t])
const ns = Array.from({ length: 161 }, (_, i) => i * 5)
const cota = (e: number) => line<number>().x(n => x(n)).y(n => y(Math.min(1, 2 * Math.exp(-2 * e * e * n))))(ns)!
const eps = [
  { e: 0.05, color: AZUL, ly: 0.55 },
  { e: 0.1, color: ROJO, ly: 0.32 },
  { e: 0.2, color: DORADO, ly: 0.045 },
]
// la etiqueta va junto al punto donde la curva cruza la altura ly
const nEn = (e: number, p: number) => Math.log(2 / p) / (2 * e * e)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Cota de Hoeffding según el tamaño de muestra">
    <Ejes :x="x" :y="y" :x-ticks="[0, 200, 400, 600, 800]" :y-ticks="[0, 0.25, 0.5, 0.75, 1]" x-titulo="tamaño de muestra n" y-titulo="cota de Hoeffding" />
    <path v-for="s in eps" :key="s.e" :d="cota(s.e)" fill="none" :stroke="s.color" stroke-width="3" />
    <g v-for="s in eps" :key="`l${s.e}`" :transform="`translate(${x(nEn(s.e, s.ly)) + 14},${y(s.ly) + 4})`">
      <line x1="0" x2="16" y1="-6" y2="-6" :stroke="s.color" stroke-width="3" />
      <text class="etiqueta" x="22" y="0" style="fill:#1A2133">ε = {{ s.e }}</text>
    </g>
  </svg>
</template>
