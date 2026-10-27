<script setup lang="ts">
// Varianza explicada acumulada de los dígitos, con 80/90/95 % marcados.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u02.json'
const G = (datos as any).digitos
const W = 720, H = 400, m = { l: 64, r: 20, t: 30, b: 60 }
const x = scaleLinear().domain([0, 64]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 1]).range([H - m.b, m.t])
const camino = line<number>().x((_, i) => x(i + 1)).y(v => y(v))([...G.acum])!
const marcas = [[0.8, G.n80], [0.9, G.n90], [0.95, G.n95]]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 16, 32, 48, 64]" :y-ticks="[0, 0.25, 0.5, 0.75, 1]" :y-formato="(v: number) => `${Math.round(v * 100)} %`" x-titulo="número de componentes m" />
    <g v-for="[p, n] in marcas" :key="p">
      <line :x1="x(n)" :x2="x(n)" :y1="y(0)" :y2="y(p)" stroke="#C9971C" stroke-width="1.5" stroke-dasharray="5 4" />
      <line :x1="m.l" :x2="x(n)" :y1="y(p)" :y2="y(p)" stroke="#C9971C" stroke-width="1.5" stroke-dasharray="5 4" />
    </g>
    <path :d="camino" fill="none" stroke="#24398A" stroke-width="3.2" />
    <g v-for="([p, n], k) in marcas" :key="`t${p}`">
      <circle :cx="x(n)" :cy="y(p)" r="6" fill="#C9971C" stroke="#fff" stroke-width="2" />
      <text :x="x(n) + 6" :y="y(0) - 12 - k * 0" text-anchor="start" style="fill: #1A2133; font-weight: 600; font-size: 17px">{{ n }}</text>
    </g>
    <text :x="x(36)" :y="y(0.5)" style="fill: #1A2133; font-size: 19px">80 % con {{ G.n80 }} componentes</text>
    <text :x="x(36)" :y="y(0.5) + 28" style="fill: #1A2133; font-size: 19px">90 % con {{ G.n90 }}</text>
    <text :x="x(36)" :y="y(0.5) + 56" style="fill: #1A2133; font-size: 19px">95 % con {{ G.n95 }}</text>
  </svg>
</template>
