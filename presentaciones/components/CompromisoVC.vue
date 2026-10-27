<script setup lang="ts">
// Esquema: al crecer d_VC, E_in baja, la penalización Ω sube y la cota de E_out tiene un mínimo.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
const W = 640, H = 380, m = { l: 40, r: 150, t: 20, b: 50 }
const ds = Array.from({ length: 81 }, (_, i) => 1 + i * 0.5)
const ein = (d: number) => 0.06 + 0.5 / (1 + 0.35 * d)
const om = (d: number) => 0.075 * Math.sqrt(d)
const x = scaleLinear().domain([1, 41]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 0.65]).range([H - m.b, m.t])
const c = (f: (d: number) => number) => line<number>().x(d => x(d)).y(d => y(f(d)))(ds)!
const dopt = ds.reduce((a, d) => ein(d) + om(d) < ein(a) + om(a) ? d : a)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <g class="eje"><line :x1="m.l" :x2="W - m.r" :y1="H - m.b" :y2="H - m.b" /><line :x1="m.l" :x2="m.l" :y1="m.t" :y2="H - m.b" />
      <text class="titulo-eje" :x="(m.l + W - m.r) / 2" :y="H - 14" text-anchor="middle">d_VC (tamaño de H)</text></g>
    <path :d="c(ein)" fill="none" stroke="#24398A" stroke-width="3.2" />
    <path :d="c(om)" fill="none" stroke="#D5433C" stroke-width="3.2" stroke-dasharray="8 6" />
    <path :d="c(d => ein(d) + om(d))" fill="none" stroke="#1A2133" stroke-width="3.2" />
    <line :x1="x(dopt)" :x2="x(dopt)" :y1="y(0)" :y2="y(ein(dopt) + om(dopt))" stroke="#C9971C" stroke-width="2" stroke-dasharray="5 4" />
    <text :x="x(41) + 10" :y="y(ein(41)) + 6" style="fill: #1A2133; font-weight: 600">E_in</text>
    <text :x="x(41) + 10" :y="y(om(41)) + 6" style="fill: #1A2133; font-weight: 600">Ω</text>
    <text :x="x(41) + 10" :y="y(ein(41) + om(41)) + 6" style="fill: #1A2133; font-weight: 600">cota de E_out</text>
    <text :x="x(dopt) + 8" :y="y(0) - 10" style="fill: #1A2133; font-size: 16px">d_VC*</text>
  </svg>
</template>
