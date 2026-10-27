<script setup lang="ts">
// Enjambre de valores SHAP de las 292 casas de prueba; color = valor del atributo (azul bajo, rojo alto).
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u05-04.json'
import { NOMBRES, divergente } from './casasAmes'
const F = (datos as any).enjambre as any[]
const W = 900, fila = 44, m = { l: 200, r: 24, t: 14, b: 62 }
const x = scaleLinear().domain([-0.34, 0.34]).range([m.l, W - m.r])
const H = m.t + F.length * fila + m.b
const ticks = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <g class="rejilla"><line v-for="t in ticks" :key="t" :x1="x(t)" :x2="x(t)" :y1="m.t" :y2="m.t + F.length * fila" /></g>
    <line :x1="x(0)" :x2="x(0)" :y1="m.t" :y2="m.t + F.length * fila" stroke="#5E6678" stroke-width="1.4" />
    <g v-for="(f, i) in F" :key="f.atr" :transform="`translate(0, ${m.t + (i + 0.5) * fila})`">
      <text :x="m.l - 14" y="6" text-anchor="end" style="fill: #1A2133">{{ NOMBRES[f.atr] }}</text>
      <circle v-for="(p, j) in f.phi" :key="j" :cx="x(p)" :cy="f.dy[j] * fila" r="3.3" :fill="divergente(f.color[j])" />
    </g>
    <g class="eje">
      <line :x1="m.l" :x2="W - m.r" :y1="m.t + F.length * fila" :y2="m.t + F.length * fila" />
      <text v-for="t in ticks" :key="t" :x="x(t)" :y="H - 36" text-anchor="middle">{{ String(t).replace('-', '−') }}</text>
      <text class="titulo-eje" :x="m.l" :y="H - 8">valor SHAP (cambio en log del precio)</text>
    </g>
    <g :transform="`translate(${W - 250}, ${H - 4})`">
      <text x="0" y="0" style="font-size: 16px">valor del atributo:</text>
      <circle cx="140" cy="-6" r="6" :fill="divergente(0)" /><text x="150" y="0" style="font-size: 16px">bajo</text>
      <circle cx="196" cy="-6" r="6" :fill="divergente(1)" /><text x="206" y="0" style="font-size: 16px">alto</text>
    </g>
  </svg>
</template>
