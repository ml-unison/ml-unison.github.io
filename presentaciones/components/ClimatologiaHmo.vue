<script setup lang="ts">
// Climatología de Hermosillo (1992–2020): cada día del año, el promedio crudo y el ajuste con 3 armónicos.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-01.json'
const props = withDefaults(defineProps<{ armonicos?: boolean }>(), { armonicos: true })
const C = (datos as any).clima
const W = 760, H = 430, m = { l: 56, r: 150, t: 30, b: 60 }
const x = scaleLinear().domain([1, 366]).range([m.l, W - m.r]), y = scaleLinear().domain([5, 43]).range([H - m.b, m.t])
const cruda = line<number>().x((_, i) => x(C.dia[i])).y(v => y(v))(C.cruda)!
const arm = line<number>().x((_, i) => x(C.dia[i])).y(v => y(v))(C.armonicos)!
const meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const inicioMes = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <Ejes :x="x" :y="y" :x-ticks="inicioMes.filter((_, k) => k % 2 === 0)" :x-formato="v => meses[inicioMes.indexOf(v)]" :y-ticks="[10, 20, 30, 40]" y-titulo="°C" x-titulo="día del año" />
    <circle v-for="(d, i) in C.pd" :key="i" :cx="x(d)" :cy="y(C.py[i])" r="1.7" fill="#6B7385" opacity="0.22" />
    <path :d="cruda" fill="none" stroke="#C9971C" stroke-width="2" />
    <path v-if="armonicos" :d="arm" fill="none" stroke="#24398A" stroke-width="5" />
    <g :transform="`translate(${W - m.r + 14}, ${y(C.cruda.at(-1)) + 40})`">
      <line x1="0" x2="18" y1="-6" y2="-6" stroke="#C9971C" stroke-width="3" />
      <text x="24" y="0" style="fill: #1A2133; font-weight: 600">promedio</text>
      <text x="24" y="22" style="fill: #1A2133; font-weight: 600">de cada día</text>
    </g>
    <g v-if="armonicos" :transform="`translate(${W - m.r + 14}, ${y(C.armonicos.at(-1)) - 34})`">
      <line x1="0" x2="18" y1="-6" y2="-6" stroke="#24398A" stroke-width="5" />
      <text x="24" y="0" style="fill: #1A2133; font-weight: 600">armónicos</text>
      <text x="24" y="22" style="fill: #1A2133; font-weight: 600">K = 3</text>
    </g>
  </svg>
</template>
