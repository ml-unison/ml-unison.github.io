<script setup lang="ts">
// Exactitud de prueba por época en Fashion-MNIST: perceptrón multicapa contra red convolucional.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-03.json'
const Fz = (datos as any).fashion
const n = Fz.cnn.length
const W = 560, H = 360, m = { l: 56, r: 210, t: 20, b: 70 }
const x = scaleLinear().domain([0.8, n + 0.2]).range([m.l, W - m.r])
const y = scaleLinear().domain([80, 92]).range([H - m.b, m.t])
const ep = Array.from({ length: n }, (_, i) => i + 1)
const series = [
  { nombre: 'red convolucional', v: Fz.cnn, p: Fz.params.cnn, color: '#24398A' },
  { nombre: 'perceptrón multicapa', v: Fz.mlp, p: Fz.params.mlp, color: '#6B7385' },
]
const d = (v: number[]) => line<number>().x((_, i) => x(i + 1)).y(a => y(100 * a))(v)!
const fmt = (p: number) => p.toLocaleString('en-US')
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="ep" :y-ticks="[80, 84, 88, 92]" x-titulo="época" y-titulo="exactitud de prueba (%)" />
    <g v-for="s in series" :key="s.nombre">
      <path :d="d(s.v)" fill="none" :stroke="s.color" stroke-width="3.5" />
      <circle v-for="(a, i) in s.v" :key="i" :cx="x(i + 1)" :cy="y(100 * a)" r="5.5" :fill="s.color" />
      <text :x="x(n) + 14" :y="y(100 * s.v[n - 1]) - 4" style="fill: #1A2133; font-weight: 600">{{ s.nombre }}</text>
      <text :x="x(n) + 14" :y="y(100 * s.v[n - 1]) + 19" style="font-size: 16px">{{ (100 * s.v[n - 1]).toFixed(1) }} % · {{ fmt(s.p) }} parám.</text>
    </g>
  </svg>
</template>
