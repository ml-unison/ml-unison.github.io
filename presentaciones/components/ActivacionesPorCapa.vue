<script setup lang="ts">
// Desviación estándar de las activaciones, capa por capa, en una red de 20 capas ReLU de 256 neuronas
// alimentada con los dígitos, para cuatro inicializaciones. Datos: herramientas/laminas/u06_01.py.
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-01.json'
const D = (datos as any).activaciones
const series = [
  { k: 'grande', n: 'N(0, 1)', c: '#D5433C' },
  { k: 'he', n: 'He', c: '#24398A' },
  { k: 'xavier', n: 'Xavier', c: '#C9971C' },
  { k: 'pequena', n: 'N(0, 0.01²)', c: '#6B7385' },
]
const W = 660, H = 440, m = { l: 74, r: 120, t: 40, b: 62 }
const x = scaleLinear().domain([1, 20]).range([m.l, W - m.r]), y = scaleLog().domain([1e-21, 1e21]).range([H - m.b, m.t])
const pot = (e: number) => e === 0 ? '1' : '10' + String(e).replace('-', '⁻').replace(/\d/g, c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c])
const camino = (v: number[]) => line<number>().x((_, i) => x(i + 1)).y(t => y(t))(v)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[1, 5, 10, 15, 20]" :y-ticks="[1e-20, 1e-10, 1, 1e10, 1e20]"
          :y-formato="v => pot(Math.round(Math.log10(v)))" x-titulo="capa" y-titulo="desviación estándar de las activaciones" />
    <g v-for="s in series" :key="s.k">
      <path :d="camino(D[s.k])" fill="none" :stroke="s.c" stroke-width="3" />
      <circle v-for="(v, i) in D[s.k]" :key="i" :cx="x(i + 1)" :cy="y(v)" r="3" :fill="s.c" />
      <text :x="x(20) + 12" :y="y(D[s.k][19]) + 6" style="fill: #1A2133; font-weight: 600">{{ s.n }}</text>
    </g>
  </svg>
</template>
