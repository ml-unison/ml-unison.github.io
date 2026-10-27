<script setup lang="ts">
// Bicicletas, 20 remuestras: ECM de prueba = varianza + (sesgo² + ruido), por modelo.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u05-02.json'
const F = (datos as any).bicis_bv
const W = 600, fila = 92, m = { l: 210, r: 30, t: 40, b: 50 }
const H = m.t + fila * F.length + m.b
const x = scaleLinear().domain([0, 7000]).range([m.l, W - m.r])
const nombre = (s: string) => s.replace(' de 50 árboles', '').replace('un árbol profundo', 'un árbol')
const fmt = (v: number) => Math.round(v).toLocaleString('es-MX')
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <g class="rejilla"><line v-for="t in [0, 2000, 4000, 6000]" :key="t" :x1="x(t)" :x2="x(t)" :y1="m.t - 10" :y2="H - m.b" /></g>
    <g class="eje"><text v-for="t in [0, 2000, 4000, 6000]" :key="t" :x="x(t)" :y="H - m.b + 26" text-anchor="middle">{{ fmt(t) }}</text></g>
    <text class="titulo-eje" :x="(m.l + W - m.r) / 2" :y="H - 6" text-anchor="middle">error cuadrático medio de prueba</text>
    <rect :x="m.l" y="6" width="16" height="16" fill="#D5433C" /><text :x="m.l + 22" y="20" style="fill: #1A2133">varianza</text>
    <rect :x="m.l + 130" y="6" width="16" height="16" fill="#C3C8D2" /><text :x="m.l + 152" y="20" style="fill: #1A2133">sesgo² + ruido</text>
    <g v-for="(f, i) in F" :key="f.modelo" :transform="`translate(0, ${m.t + i * fila + 14})`">
      <text :x="m.l - 14" y="34" text-anchor="end" style="fill: #1A2133; font-weight: 600; font-size: 20px">{{ nombre(f.modelo) }}</text>
      <rect :x="x(0)" y="8" :width="x(f.varianza) - x(0)" height="44" fill="#D5433C" />
      <rect :x="x(f.varianza)" y="8" :width="x(f.varianza + f.resto) - x(f.varianza)" height="44" fill="#C3C8D2" />
      <text v-if="x(f.varianza) - x(0) > 70" :x="x(0) + 8" y="37" style="fill: #fff; font-weight: 700">{{ fmt(f.varianza) }}</text>
      <text v-else :x="x(f.varianza) + 8" y="37" style="fill: #B23028; font-weight: 700">{{ fmt(f.varianza) }}</text>
      <text :x="x(f.varianza + f.resto) - 8" y="37" text-anchor="end" style="fill: #1A2133">{{ fmt(f.resto) }}</text>
      <text :x="x(f.varianza + f.resto) + 8" y="37" style="fill: #1A2133; font-weight: 600">= {{ fmt(f.ecm) }}</text>
    </g>
  </svg>
</template>
