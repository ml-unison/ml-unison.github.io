<script setup lang="ts">
// Probabilidad de daño (al menos una junta) según la regresión logística, con banda bootstrap del 90 %.
import { scaleLinear } from 'd3-scale'
import { area, line } from 'd3-shape'
import datos from '../datos/u04.json'
const C = (datos as any).challenger
const W = 700, H = 400, m = { l: 64, r: 30, t: 36, b: 64 }
const x = scaleLinear().domain([28, 84]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 1]).range([H - m.b, m.t])
const banda = area<number>().x((_, i) => x(C.t[i])).y0((_, i) => y(C.p05[i])).y1((_, i) => y(C.p95[i]))(C.t)!
const curva = line<number>().x((_, i) => x(C.t[i])).y(v => y(v))(C.p)!
const jit = (i: number) => ((i * 7919) % 11 - 5) * 0.012
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[30, 40, 50, 60, 70, 80]" :y-ticks="[0, 0.25, 0.5, 0.75, 1]" x-titulo="temperatura al despegue (°F)" y-titulo="P(al menos una junta dañada)" />
    <path :d="banda" fill="#C0CEE3" opacity="0.7" />
    <path :d="curva" fill="none" stroke="#24398A" stroke-width="3.5" />
    <circle v-for="(t, i) in C.temp" :key="i" :cx="x(t)" :cy="y(C.danadas[i] > 0 ? 1 - Math.abs(jit(i)) : Math.abs(jit(i)))" r="6" :fill="C.danadas[i] > 0 ? '#D5433C' : '#24398A'" stroke="#fff" stroke-width="1.5" />
    <line :x1="x(31)" :x2="x(31)" :y1="m.t" :y2="H - m.b" stroke="#D5433C" stroke-width="2" stroke-dasharray="6 5" />
    <text :x="x(31) + 8" :y="y(0.62)" style="fill: #1A2133; font-weight: 600">31 °F</text>
  </svg>
</template>
