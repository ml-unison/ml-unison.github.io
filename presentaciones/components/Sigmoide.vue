<script setup lang="ts">
// La función logística σ(z) = 1 / (1 + e^{−z}).
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
const W = 460, H = 330, m = { l: 50, r: 16, t: 24, b: 76 }
const x = scaleLinear().domain([-6, 6]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 1]).range([H - m.b, m.t])
const s = Array.from({ length: 121 }, (_, i) => -6 + i * 0.1)
const d = line<number>().x(v => x(v)).y(v => y(1 / (1 + Math.exp(-v))))(s)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[-6, -3, 0, 3, 6]" :y-ticks="[0, 0.5, 1]" x-titulo="z = wᵀx + b" />
    <path :d="d" fill="none" stroke="#24398A" stroke-width="3.5" />
    <text :x="x(1.2)" :y="y(0.35)" style="fill: #1A2133; font-weight: 600; font-size: 20px">σ(z) = 1 / (1 + e⁻ᶻ)</text>
  </svg>
</template>
