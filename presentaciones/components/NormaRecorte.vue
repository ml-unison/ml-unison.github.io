<script setup lang="ts">
// Norma del gradiente (antes del recorte) en cada paso del entrenamiento de la LSTM.
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-04.json'
const R = (datos as any).recorte
const W = 620, H = 380, m = { l: 62, r: 20, t: 30, b: 70 }
const x = scaleLinear().domain([0, R.normas.length]).range([m.l, W - m.r])
const y = scaleLog().domain([0.08, 2]).range([H - m.b, m.t])
const d = line<number>().x((_, i) => x(i + 1)).y(v => y(v))(R.normas)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 300, 600, 900, 1200]" :y-ticks="[0.1, 0.2, 0.5, 1, 2]" x-titulo="paso de entrenamiento" y-titulo="‖g‖ antes de recortar" />
    <path :d="d" fill="none" stroke="#24398A" stroke-width="1.3" />
    <line :x1="m.l" :x2="W - m.r" :y1="y(1)" :y2="y(1)" stroke="#D5433C" stroke-width="2" stroke-dasharray="7 5" />
    <text :x="W - m.r" :y="y(1) - 10" text-anchor="end" class="etiqueta" style="fill: #1A2133">umbral γ = 1</text>
  </svg>
</template>
