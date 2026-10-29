<script setup lang="ts">
// Ruido del gradiente de un minilote, E‖g_B − g‖², contra B (escala log-log), con la señal ‖g‖².
import { scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-02.json'
const R = (datos as any).ruido
const W = 600, H = 350, m = { l: 70, r: 20, t: 40, b: 64 }
const x = scaleLog().domain([0.8, 1300]).range([m.l, W - m.r])
const y = scaleLog().domain([0.08, 400]).range([H - m.b, m.t])
const N = 10000
const bb = Array.from({ length: 80 }, (_, i) => Math.pow(10, i / 79 * Math.log10(1024)))
const teoria = line<number>().x(b => x(b)).y(b => y(R.sigma2 / b * (N - b) / (N - 1)))(bb)!
const fy = (v: number) => v >= 1 ? String(v) : String(v).replace('.', '.')
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[1, 4, 16, 64, 256, 1024]" :y-ticks="[0.1, 1, 10, 100]" :y-formato="fy" :rejilla-x="false"
          x-titulo="tamaño del minilote |B|" y-titulo="ruido: E‖g_B − g‖²" />
    <path :d="teoria" fill="none" stroke="#6B7385" stroke-width="2" stroke-dasharray="7 5" />
    <line :x1="m.l" :x2="W - m.r" :y1="y(R.g2)" :y2="y(R.g2)" stroke="#D5433C" stroke-width="2.5" />
    <text :x="x(1.05)" :y="y(R.g2) - 10" style="fill: #1A2133; font-weight: 600">señal: ‖g‖² = {{ R.g2.toFixed(2) }}</text>
    <text :x="x(40)" :y="y(R.sigma2 / 40) - 16" style="fill: #5E6678">σ²/|B|</text>
    <circle v-for="(b, i) in R.B" :key="b" :cx="x(b)" :cy="y(R.e[i])" r="8" fill="#24398A" stroke="#fff" stroke-width="2" />
  </svg>
</template>
