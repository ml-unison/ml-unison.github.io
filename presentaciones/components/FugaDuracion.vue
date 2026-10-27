<script setup lang="ts">
// Mercadeo bancario: distribución de la duración de la llamada según la respuesta del cliente.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u05-03.json'
const Hd = (datos as any).banco.hist
const W = 620, H = 380, m = { l: 56, r: 16, t: 30, b: 60 }
const x = scaleLinear().domain([0, 1470]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 4.2]).range([H - m.b, m.t])
const barras = (v: number[]) => v.map((d, i) => ({ x: x(Hd.bordes[i]), w: x(Hd.bordes[i + 1]) - x(Hd.bordes[i]) - 0.5, y: y(d), h: y(0) - y(d) }))
const no = barras(Hd.no), si = barras(Hd.si)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 300, 600, 900, 1200]" :y-ticks="[0, 1, 2, 3, 4]" x-titulo="duración de la llamada (segundos)" y-titulo="densidad (por mil, por segundo)" />
    <rect v-for="(b, i) in no" :key="`n${i}`" v-bind="{ x: b.x, y: b.y, width: b.w, height: b.h }" fill="#24398A" opacity="0.55" />
    <rect v-for="(b, i) in si" :key="`s${i}`" v-bind="{ x: b.x, y: b.y, width: b.w, height: b.h }" fill="#D5433C" opacity="0.55" />
    <text :x="x(260)" :y="y(3.2)" class="etiqueta" style="fill: #1A2133">no contrata</text>
    <line :x1="x(250)" :x2="x(190)" :y1="y(3.15)" :y2="y(2.6)" stroke="#24398A" stroke-width="1.5" />
    <text :x="x(700)" :y="y(1.45)" class="etiqueta" style="fill: #1A2133">contrata</text>
    <line :x1="x(690)" :x2="x(640)" :y1="y(1.38)" :y2="y(0.95)" stroke="#D5433C" stroke-width="1.5" />
  </svg>
</template>
