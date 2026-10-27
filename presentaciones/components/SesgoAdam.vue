<script setup lang="ts">
// Adam sin corrección de sesgo: el paso, con gradiente constante, es η(1 − β₁ᵗ)/√(1 − β₂ᵗ) en lugar de η.
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-02.json'
const A = (datos as any).adam
const W = 560, H = 360, m = { l: 56, r: 20, t: 36, b: 66 }
const x = scaleLog().domain([1, 3000]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 7]).range([H - m.b, m.t])
const d = line<number>().x((_, i) => x(A.t[i])).y(v => y(v))(A.f)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[1, 10, 100, 1000]" :y-ticks="[0, 1, 2, 3, 4, 5, 6, 7]" x-titulo="paso t" y-titulo="tamaño del paso / η" />
    <line :x1="m.l" :x2="W - m.r" :y1="y(1)" :y2="y(1)" stroke="#24398A" stroke-width="3" />
    <path :d="d" fill="none" stroke="#D5433C" stroke-width="3.5" />
    <circle :cx="x(A.tmax)" :cy="y(A.max)" r="6" fill="#D5433C" />
    <text :x="x(A.tmax) + 14" :y="y(A.max) + 4" style="fill: #1A2133; font-weight: 600">{{ A.max.toFixed(1) }} veces en t = {{ A.tmax }}</text>
    <text :x="x(1.1)" :y="y(1) - 12" style="fill: #24398A; font-weight: 600">con corrección</text>
    <text :x="x(150)" :y="y(3.6)" style="fill: #D5433C; font-weight: 600">sin corrección</text>
  </svg>
</template>
