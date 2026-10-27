<script setup lang="ts">
// Residuos de M4 (regresión lineal) contra el valor ajustado, en 1,800 horas de prueba.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u04.json'
const R = (datos as any).residuos
const W = 660, H = 420, m = { l: 70, r: 20, t: 36, b: 64 }
const x = scaleLinear().domain([-150, 800]).range([m.l, W - m.r]), y = scaleLinear().domain([-400, 500]).range([H - m.b, m.t]).clamp(true)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <rect :x="x(-150)" :y="m.t" :width="x(0) - x(-150)" :height="H - m.t - m.b" fill="#FCEFEE" />
    <text :x="x(-140)" :y="m.t + 22" style="fill: #1A2133; font-weight: 600; font-size: 17px">predice &lt; 0</text>
    <Ejes :x="x" :y="y" :x-ticks="[0, 200, 400, 600, 800]" :y-ticks="[-400, -200, 0, 200, 400]" x-titulo="valor ajustado ŷ" y-titulo="residuo y − ŷ" />
    <line :x1="x(-150)" :x2="x(800)" :y1="y(0)" :y2="y(0)" stroke="#1A2133" stroke-width="1.5" />
    <circle v-for="(v, i) in R.ajustado" :key="i" :cx="x(v)" :cy="y(R.residuo[i])" r="3" fill="#24398A" opacity="0.3" />
  </svg>
</template>
