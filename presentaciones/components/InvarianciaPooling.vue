<script setup lang="ts">
// Cambio relativo de la representación al mover la imagen un pixel, a lo largo de la red (pesos al azar).
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u06-03.json'
const I = (datos as any).invariancia as { etapa: string, cambio: number }[]
const W = 560, H = 300, m = { l: 150, r: 70, t: 10, b: 60 }
const x = scaleLinear().domain([0, 0.35]).range([m.l, W - m.r])
const banda = (H - m.t - m.b) / I.length
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <line v-for="t in [0, 0.1, 0.2, 0.3]" :key="t" :x1="x(t)" :x2="x(t)" :y1="m.t" :y2="H - m.b" stroke="#E2E5EC" />
    <text v-for="t in [0, 0.1, 0.2, 0.3]" :key="`t${t}`" :x="x(t)" :y="H - m.b + 26" text-anchor="middle">{{ t }}</text>
    <text :x="(m.l + W - m.r) / 2" :y="H - 6" text-anchor="middle" class="titulo-eje">‖f(Tx) − f(x)‖ / ‖f(x)‖</text>
    <g v-for="(e, k) in I" :key="e.etapa">
      <rect :x="x(0)" :y="m.t + k * banda + 8" :width="x(e.cambio) - x(0)" :height="banda - 16"
            :fill="e.etapa.includes('pool') ? '#24398A' : '#8DA3C9'" rx="3" />
      <text :x="m.l - 12" :y="m.t + (k + 0.5) * banda + 6" text-anchor="end" style="fill: #1A2133; font-weight: 600">{{ e.etapa }}</text>
      <text :x="x(e.cambio) + 8" :y="m.t + (k + 0.5) * banda + 6" style="fill: #1A2133">{{ e.cambio.toFixed(2) }}</text>
    </g>
  </svg>
</template>
