<script setup lang="ts">
// Cuatro calendarios para la tasa de aprendizaje, en 30 épocas.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-02.json'
const C = (datos as any).calendarios
const W = 560, H = 360, m = { l: 56, r: 20, t: 36, b: 66 }
const x = scaleLinear().domain([0, 30]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 1.05]).range([H - m.b, m.t])
const d = (k: string) => line<number>().x((_, i) => x(C.epoca[i])).y(v => y(v))(C[k])!
const series = [
  { k: 'constante', c: '#6B7385', w: 3, et: [21, 1.0], dy: -10 },
  { k: 'escalón', c: '#C9971C', w: 3, et: [15.6, 0.1], dy: -10 },
  { k: 'coseno', c: '#24398A', w: 3.5, et: [11, 0.78], dy: -10 },
  { k: 'calentamiento + coseno', c: '#D5433C', w: 3, d: '4 5', et: [2.4, 0.3], dy: 0 },
]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 10, 20, 30]" :y-ticks="[0, 0.5, 1]" x-titulo="época" y-titulo="η / η₀" />
    <path v-for="s in series" :key="s.k" :d="d(s.k)" fill="none" :stroke="s.c" :stroke-width="s.w" :stroke-dasharray="s.d" />
    <text v-for="s in series" :key="`t${s.k}`" :x="x(s.et[0])" :y="y(s.et[1]) + s.dy" style="fill: #1A2133; font-weight: 600; font-size: 18px">
      {{ s.k === 'calentamiento + coseno' ? 'calentamiento' : s.k }}
    </text>
  </svg>
</template>
