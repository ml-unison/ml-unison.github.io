<script setup lang="ts">
// Perplejidad de validación de la LSTM y del transformer durante el entrenamiento, con las líneas base de n-gramas.
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-04.json'
const C = (datos as any).curvas
const W = 640, H = 400, m = { l: 56, r: 120, t: 30, b: 70 }
const x = scaleLinear().domain([0, 1500]).range([m.l, W - m.r])
const y = scaleLog().domain([3.4, 12]).range([H - m.b, m.t])
const l = (s: any) => line<number>().x((_, i) => x(s.paso[i])).y(v => y(v))(s.va)!
const series = [{ s: C.lstm, c: '#24398A', t: 'LSTM' }, { s: C.tf, c: '#D5433C', t: 'transformer' }]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 500, 1000, 1500]" :y-ticks="[4, 5, 6, 8, 10]" x-titulo="paso (32 ventanas de 128 letras)" y-titulo="perplejidad de validación" />
    <line :x1="m.l" :x2="W - m.r" :y1="y(C.bigrama)" :y2="y(C.bigrama)" stroke="#6B7385" stroke-width="2" stroke-dasharray="3 5" />
    <line :x1="m.l" :x2="W - m.r" :y1="y(C.ngrama)" :y2="y(C.ngrama)" stroke="#C9971C" stroke-width="2.2" stroke-dasharray="8 5" />
    <text :x="W - m.r + 8" :y="y(C.bigrama) + 6" style="fill: #1A2133">bigrama</text>
    <text :x="W - m.r + 8" :y="y(C.ngrama) + 6" style="fill: #1A2133">5-grama</text>
    <g v-for="s in series" :key="s.t">
      <path :d="l(s.s)" fill="none" :stroke="s.c" stroke-width="3" />
      <circle v-for="(v, i) in s.s.va" :key="i" :cx="x(s.s.paso[i])" :cy="y(v)" r="4" :fill="s.c" />
    </g>
    <text :x="x(700)" :y="y(C.lstm.va[6]) + 30" text-anchor="middle" class="etiqueta" style="fill: #24398A">LSTM</text>
    <text :x="x(300) + 12" :y="y(C.tf.va[2]) - 8" class="etiqueta" style="fill: #D5433C">transformer</text>
  </svg>
</template>
