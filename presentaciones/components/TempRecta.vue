<script setup lang="ts">
// Rentas por hora contra temperatura (1,500 horas de entrenamiento) y la recta de mínimos cuadrados.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u04.json'
const T = (datos as any).temp
const W = 620, H = 420, m = { l: 64, r: 20, t: 36, b: 64 }
const x = scaleLinear().domain([0, 42]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 1000]).range([H - m.b, m.t])
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 10, 20, 30, 40]" :y-ticks="[0, 250, 500, 750, 1000]" x-titulo="temperatura (°C)" y-titulo="bicicletas en la hora" />
    <circle v-for="(v, i) in T.x" :key="i" :cx="x(v)" :cy="y(T.y[i])" r="3" fill="#24398A" opacity="0.28" />
    <line :x1="x(0)" :y1="y(T.w0)" :x2="x(42)" :y2="y(T.w0 + 42 * T.w1)" stroke="#D5433C" stroke-width="3.5" />
    <text :x="x(30)" :y="y(T.w0 + 42 * T.w1) - 12" style="fill: #1A2133; font-weight: 600">ŷ = {{ T.w0.toFixed(1) }} + {{ T.w1.toFixed(2) }}·temp</text>
  </svg>
</template>
