<script setup lang="ts">
// Línea del tiempo del perceptrón a la retropropagación (1958–2015), con las referencias de la clase.
import { scaleLinear } from 'd3-scale'
const W = 1120, H = 330, ej = 165
const x = scaleLinear().domain([1955, 2016]).range([30, W - 30])
const hitos = [
  { a: 1958, t: ['Rosenblatt:', 'el perceptrón', 'aprende de ejemplos'], arriba: true, c: '#24398A' },
  { a: 1969, t: ['Minsky y Papert:', 'los límites de', 'una sola capa'], arriba: false, c: '#D5433C' },
  { a: 1986, t: ['Rumelhart, Hinton', 'y Williams:', 'retropropagación'], arriba: true, c: '#24398A' },
  { a: 1989, t: ['Cybenko: aproximación', 'universal'], arriba: false, c: '#C9971C' },
  { a: 2010, t: ['Glorot y Bengio:', 'inicialización'], arriba: true, c: '#C9971C' },
  { a: 2012, t: ['AlexNet gana', 'ImageNet (tema 3)'], arriba: false, c: '#24398A' },
]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <line :x1="x(1955)" :x2="x(2016)" :y1="ej" :y2="ej" stroke="#8A91A0" stroke-width="2" />
    <g v-for="a in [1960, 1970, 1980, 1990, 2000, 2010]" :key="a">
      <line :x1="x(a)" :x2="x(a)" :y1="ej - 5" :y2="ej + 5" stroke="#8A91A0" stroke-width="2" />
    </g>
    <g v-for="h in hitos" :key="h.a">
      <line :x1="x(h.a)" :x2="x(h.a)" :y1="ej" :y2="h.arriba ? ej - 44 : ej + 44" :stroke="h.c" stroke-width="2" />
      <circle :cx="x(h.a)" :cy="ej" r="8" :fill="h.c" stroke="#fff" stroke-width="2" />
      <text :x="x(h.a)" :y="h.arriba ? ej - 52 - 22 * (h.t.length - 1) - 26 : ej + 70" text-anchor="middle"
            style="fill: #20305A; font-weight: 700; font-size: 20px">{{ h.a }}</text>
      <text v-for="(l, i) in h.t" :key="i" :x="x(h.a)" :y="h.arriba ? ej - 52 - 22 * (h.t.length - 1 - i) : ej + 94 + 22 * i"
            text-anchor="middle" style="fill: #1A2133; font-size: 17px">{{ l }}</text>
    </g>
  </svg>
</template>
