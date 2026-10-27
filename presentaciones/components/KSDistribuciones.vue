<script setup lang="ts">
// Funciones de distribución empíricas de la temperatura de marzo (2011 y 2012) y el estadístico D de Kolmogorov–Smirnov.
import { scaleLinear } from 'd3-scale'
import { line, curveStepAfter } from 'd3-shape'
import d from '../datos/u08-02.json'
const K = (d as any).ks_marzo
const props = withDefaults(defineProps<{ conD?: boolean }>(), { conD: true })
const W = 640, H = 400, m = { l: 62, r: 20, t: 34, b: 62 }
const x = scaleLinear().domain([0, 32]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 1]).range([H - m.b, m.t])
const escal = (F: number[]) => line<number>().curve(curveStepAfter).x((_, i) => x(K.x[i])).y(v => y(v))(F)!
const i = K.x.indexOf(K.xmax)
const fa = K.F2011[i], fc = K.F2012[i]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <Ejes :x="x" :y="y" :x-ticks="[0, 5, 10, 15, 20, 25, 30]" :y-ticks="[0, 0.25, 0.5, 0.75, 1]" x-titulo="temperatura en marzo (°C)" y-titulo="proporción de horas con temperatura ≤ x" />
    <path :d="escal(K.F2011)" fill="none" stroke="#24398A" stroke-width="3" />
    <path :d="escal(K.F2012)" fill="none" stroke="#D5433C" stroke-width="3" />
    <text :x="x(8.5)" :y="y(0.62)" class="etiqueta" style="fill:#24398A" text-anchor="end">2011 · F₁</text>
    <text :x="x(21.5)" :y="y(0.55)" class="etiqueta" style="fill:#D5433C">2012 · F₂</text>
    <g v-if="conD">
      <line :x1="x(K.xmax)" :x2="x(K.xmax)" :y1="y(fa)" :y2="y(fc)" stroke="#1A2133" stroke-width="2.4" />
      <line :x1="x(K.xmax) - 7" :x2="x(K.xmax) + 7" :y1="y(fa)" :y2="y(fa)" stroke="#1A2133" stroke-width="2.4" />
      <line :x1="x(K.xmax) - 7" :x2="x(K.xmax) + 7" :y1="y(fc)" :y2="y(fc)" stroke="#1A2133" stroke-width="2.4" />
      <text :x="x(K.xmax) + 4" :y="y(fc) + 34" style="fill:#1A2133; font-weight: 650; font-size: 21px">D = {{ K.D.toFixed(3) }}</text>
      <text :x="x(K.xmax) + 4" :y="y(fc) + 56" style="font-size: 16px">en {{ K.xmax }} °C</text>
    </g>
  </svg>
</template>
