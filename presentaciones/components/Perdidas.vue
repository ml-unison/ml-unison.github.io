<script setup lang="ts">
// Pérdidas como función del margen funcional y·s: 0-1, bisagra, logística (en base 2) y cuadrática.
import { scaleLinear } from 'd3-scale'
const W = 640, H = 400
const x = scaleLinear().domain([-2, 3]).range([70, 470]), y = scaleLinear().domain([0, 4]).range([H - 60, 30])
const s = Array.from({ length: 251 }, (_, i) => -2 + i * 0.02)
const series = {
  'cero-uno': s.map(v => (v <= 0 ? 1 : 0)),
  'bisagra': s.map(v => Math.max(0, 1 - v)),
  'logística': s.map(v => Math.log2(1 + Math.exp(-v))),
  'cuadrática': s.map(v => (1 - v) ** 2),
}
</script>

<template>
  <LineasLog :x="s" :series="series" :y-log="false" :y-dominio="[0, 4]" :y-ticks="[0, 1, 2, 3, 4]" :x-dominio="[-2, 3]" :x-ticks="[-2, -1, 0, 1, 2, 3]"
             x-titulo="y · s   (s = wᵀx + b)" y-titulo="pérdida" :colores="['#6B7385', '#C9971C', '#24398A', '#D5433C']" :punteadas="['cuadrática']" :ancho="680" :alto="400" :derecha="150" />
</template>
