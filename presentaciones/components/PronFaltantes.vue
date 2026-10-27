<script setup lang="ts">
// Horas faltantes en la tabla de bicicletas, por hora del día.
import { scaleLinear, scaleBand } from 'd3-scale'
import datos from '../datos/u07-03.json'
const F = (datos as any).faltantes
const W = 620, H = 340, m = { l: 56, r: 10, t: 36, b: 64 }
const x = scaleBand<number>().domain(Array.from({ length: 24 }, (_, i) => i)).range([m.l, W - m.r]).padding(0.18)
const y = scaleLinear().domain([0, 36]).range([H - m.b, m.t])
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <g class="rejilla"><line v-for="t in [0, 10, 20, 30]" :key="t" :x1="m.l" :x2="W - m.r" :y1="y(t)" :y2="y(t)" /></g>
    <rect v-for="(v, i) in F.por_hora" :key="i" :x="x(i)" :y="y(v)" :width="x.bandwidth()" :height="y(0) - y(v)" :fill="i >= 2 && i <= 5 ? '#24398A' : '#8DA3C9'" />
    <g class="eje">
      <line :x1="m.l" :x2="W - m.r" :y1="y(0)" :y2="y(0)" />
      <text v-for="i in [0, 3, 6, 9, 12, 15, 18, 21]" :key="i" :x="x(i)! + x.bandwidth() / 2" :y="y(0) + 26" text-anchor="middle">{{ i }}</text>
      <text v-for="t in [0, 10, 20, 30]" :key="`y${t}`" :x="m.l - 10" :y="y(t) + 6" text-anchor="end">{{ t }}</text>
      <text class="titulo-eje" :x="(m.l + W - m.r) / 2" :y="H - 8" text-anchor="middle">hora del día</text>
      <text class="titulo-eje" :x="m.l - 50" :y="m.t - 16">horas faltantes (de {{ F.n }})</text>
    </g>
  </svg>
</template>
