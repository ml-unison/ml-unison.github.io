<script setup lang="ts">
// 200 réplicas bootstrap de los coeficientes de temp y sensacion ajustados juntos.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u04.json'
const C = (datos as any).colineal
const W = 520, H = 420, m = { l: 64, r: 20, t: 36, b: 64 }
const x = scaleLinear().domain([3, 13]).range([m.l, W - m.r]), y = scaleLinear().domain([-4, 6]).range([H - m.b, m.t])
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[4, 6, 8, 10, 12]" :y-ticks="[-4, -2, 0, 2, 4, 6]" x-titulo="coeficiente de temp" y-titulo="coeficiente de sensacion" />
    <circle v-for="(v, i) in C.temp" :key="i" :cx="x(v)" :cy="y(C.sens[i])" r="4.5" fill="#24398A" opacity="0.55" />
    <line :x1="x(Math.min(...C.solo))" :x2="x(Math.max(...C.solo))" :y1="y(0)" :y2="y(0)" stroke="#D5433C" stroke-width="7" stroke-linecap="round" />
    <text :x="x(9.2)" :y="y(-1.2)" text-anchor="middle" style="fill: #1A2133; font-weight: 600">temp sola</text>
  </svg>
</template>
