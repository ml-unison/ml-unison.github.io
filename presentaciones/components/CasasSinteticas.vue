<script setup lang="ts">
// Casas reales (área, calidad) y las casas sintéticas que la dependencia parcial evalúa en 250 y 300 m².
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u05-04.json'
const E = (datos as any).extrapolacion
const props = withDefaults(defineProps<{ sinteticas?: boolean }>(), { sinteticas: true })
const W = 700, H = 430, m = { l: 56, r: 20, t: 34, b: 64 }
const x = scaleLinear().domain([20, 400]).range([m.l, W - m.r]), y = scaleLinear().domain([0.4, 10.6]).range([H - m.b, m.t])
const sac = (i: number, s: number) => (((i * 0.6180339887 + s) % 1) - 0.5)
const reales = E.area.map((a: number, i: number) => [x(a), y(E.calidad[i] + 0.5 * sac(i, 0.13))])
const sint = [250, 300].flatMap(v => E.calidad.map((c: number, i: number) => [x(v + 6 * sac(i, 0.71)), y(c + 0.5 * sac(i, 0.13))]))
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[50, 100, 150, 200, 250, 300, 350]" :y-ticks="[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]" x-titulo="área habitable (m²)" y-titulo="calidad general" />
    <circle v-for="(p, i) in reales" :key="i" :cx="p[0]" :cy="p[1]" r="3.4" fill="#24398A" opacity="0.5" />
    <g v-if="sinteticas">
      <circle v-for="(p, i) in sint" :key="`s${i}`" :cx="p[0]" :cy="p[1]" r="2.6" fill="#D5433C" opacity="0.28" />
      <rect :x="x(236)" :y="y(5.5)" :width="x(314) - x(236)" :height="y(0.5) - y(5.5)" fill="none" stroke="#B23028" stroke-width="2" stroke-dasharray="6 4" rx="6" />
      <text :x="x(320)" :y="y(3.4)" style="fill: #B23028; font-weight: 650; font-size: 20px">casas que</text>
      <text :x="x(320)" :y="y(3.4) + 24" style="fill: #B23028; font-weight: 650; font-size: 20px">no existen</text>
    </g>
  </svg>
</template>
