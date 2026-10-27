<script setup lang="ts">
// Diagrama de confiabilidad: probabilidad predicha contra frecuencia observada (deciles).
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u03.json'
const K = (datos as any).calibracion
const W = 560, H = 460, m = { l: 64, r: 150, t: 20, b: 64 }
const x = scaleLinear().domain([0, 0.85]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 0.85]).range([H - m.b, m.t])
const mods = [['logistica', '#D5433C', 'logística'], ['bosque', '#24398A', 'bosque'], ['bosque_iso', '#C9971C', 'bosque calibrado']]
const c = (k: string) => line<number>().x((_, i) => x(K[k].predicha[i])).y(v => y(v))(K[k].observada)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 0.2, 0.4, 0.6, 0.8]" :y-ticks="[0, 0.2, 0.4, 0.6, 0.8]" x-titulo="probabilidad predicha (promedio del decil)" y-titulo="fracción que incumplió" />
    <line :x1="x(0)" :y1="y(0)" :x2="x(0.85)" :y2="y(0.85)" stroke="#8A91A0" stroke-width="1.5" stroke-dasharray="5 4" />
    <g v-for="([k, col], j) in mods" :key="k">
      <path :d="c(k)" fill="none" :stroke="col" stroke-width="2.8" />
      <circle v-for="(v, i) in K[k].observada" :key="i" :cx="x(K[k].predicha[i])" :cy="y(v)" r="5" :fill="col" stroke="#fff" stroke-width="1.5" />
      <line :x1="W - m.r + 14" :x2="W - m.r + 32" :y1="m.t + 20 + j * 30" :y2="m.t + 20 + j * 30" :stroke="col" stroke-width="3" />
      <text :x="W - m.r + 38" :y="m.t + 26 + j * 30" style="fill: #1A2133; font-size: 16px; font-weight: 600">{{ mods[j][2] }}</text>
    </g>
    <text :x="x(0.62)" :y="y(0.55)" style="font-size: 15px" :transform="`rotate(-38 ${x(0.62)} ${y(0.55)})`">calibración perfecta</text>
  </svg>
</template>
