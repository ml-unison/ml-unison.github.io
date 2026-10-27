<script setup lang="ts">
// Precio contra área habitable, con las cuatro casas de más de 4,000 pies² resaltadas.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u01.json'
const D = (datos as any).casas.dispersion
const W = 760, H = 420, m = { l: 60, r: 20, t: 30, b: 60 }
const x = scaleLinear().domain([0, 6000]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 800]).range([H - m.b, m.t])
const pts = D.area.map((a: number, i: number) => ({ a, p: D.precio[i], raro: a > 4000 }))
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <Ejes :x="x" :y="y" :x-ticks="[0, 2000, 4000, 6000]" :y-ticks="[0, 200, 400, 600, 800]" x-titulo="área habitable (pies²)" y-titulo="miles de dólares" />
    <circle v-for="(p, i) in pts.filter((q: any) => !q.raro)" :key="i" :cx="x(p.a)" :cy="y(p.p)" r="3.5" fill="#24398A" opacity="0.45" />
    <circle v-for="(p, i) in pts.filter((q: any) => q.raro)" :key="`r${i}`" :cx="x(p.a)" :cy="y(p.p)" r="9" fill="#D5433C" stroke="#fff" stroke-width="2" />
    <text :x="x(4700) + 16" :y="y(185) + 34" style="font-size: 16px; fill: #B23028">ventas parciales</text>
    <text :x="x(4400) - 14" :y="y(750) + 6" text-anchor="end" style="font-size: 16px; fill: #5E6678">casas grandes y caras</text>
  </svg>
</template>
