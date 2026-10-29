<script setup lang="ts">
// Norma del gradiente de la pérdida respecto de los pesos de cada capa oculta (red de 10 capas de 64,
// en la inicialización, sobre 256 dígitos): sigmoide con Xavier contra ReLU con He. Datos: herramientas/laminas/u06_01.py.
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-01.json'
const D = (datos as any).gradientes
const W = 640, H = 420, m = { l: 74, r: 24, t: 40, b: 66 }
const x = scaleLinear().domain([1, 10]).range([m.l, W - m.r]), y = scaleLog().domain([3e-7, 1]).range([H - m.b, m.t])
const pot = (e: number) => e === 0 ? '1' : '10' + String(e).replace('-', '⁻').replace(/\d/g, c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c])
const camino = (v: number[]) => line<number>().x((_, i) => x(i + 1)).y(t => y(t))(v)!
const series = [{ k: 'relu', n: 'ReLU, He', c: '#24398A', i: 0, dy: -16 }, { k: 'sigmoide', n: 'sigmoide, Xavier', c: '#D5433C', i: 4, dy: 34 }]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]" :y-ticks="[1e-6, 1e-4, 1e-2, 1]"
          :y-formato="v => pot(Math.round(Math.log10(v)))" x-titulo="capa oculta (1 = junto a la entrada)" y-titulo="‖∂L/∂W‖ de la capa" />
    <g v-for="s in series" :key="s.k">
      <path :d="camino(D[s.k])" fill="none" :stroke="s.c" stroke-width="3" />
      <circle v-for="(v, i) in D[s.k]" :key="i" :cx="x(i + 1)" :cy="y(v)" r="4" :fill="s.c" />
      <text :x="x(s.i + 1) + 6" :y="y(D[s.k][s.i]) + s.dy" style="fill: #1A2133; font-weight: 600">{{ s.n }}</text>
    </g>
  </svg>
</template>
