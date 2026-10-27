<script setup lang="ts">
// Las palabras con mayor y menor log p(w|spam) − log p(w|normal) según Bayes ingenuo.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u04.json'
const S = (datos as any).sms
const filas = [...S.spam.map((p: any[]) => ({ w: p[0], v: p[1] })), ...S.normal.slice().reverse().map((p: any[]) => ({ w: p[0], v: p[1] }))]
const W = 560, fh = 18, H = filas.length * fh + 96, cx = W / 2
const x = scaleLinear().domain([-6, 6]).range([20, W - 20])
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <text :x="x(3)" y="20" text-anchor="middle" style="fill: #1A2133; font-weight: 700">más probable en spam</text>
    <text :x="x(-3)" :y="H - 36" text-anchor="middle" style="fill: #1A2133; font-weight: 700">más probable en mensajes normales</text>
    <g v-for="(f, i) in filas" :key="f.w">
      <rect :x="Math.min(x(0), x(f.v))" :y="32 + i * fh" :width="Math.abs(x(f.v) - x(0))" :height="fh - 2" :fill="f.v > 0 ? '#D5433C' : '#24398A'" rx="2" />
      <text :x="f.v > 0 ? x(0) - 8 : x(0) + 8" :y="32 + i * fh + 13" :text-anchor="f.v > 0 ? 'end' : 'start'" style="fill: #1A2133; font-size: 15px; font-family: var(--f-codigo)">{{ f.w }}</text>
    </g>
    <line :x1="x(0)" :x2="x(0)" y1="28" :y2="34 + filas.length * fh" stroke="#1A2133" />
    <g class="eje"><text v-for="t in [-4, -2, 2, 4]" :key="t" :x="x(t)" :y="H - 8" text-anchor="middle" style="font-size: 15px">{{ t > 0 ? '+' : '−' }}{{ Math.abs(t) }}</text></g>
  </svg>
</template>
