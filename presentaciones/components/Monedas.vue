<script setup lang="ts">
// 1,000 monedas justas, 10 lanzamientos cada una: distribución de ν para la primera, una al azar y la de menos caras.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u03.json'
const M = (datos as any).monedas
const paneles = [['primera', 'la primera moneda'], ['azar', 'una moneda al azar'], ['minima', 'la de menos caras']]
const W = 360, H = 250, m = { l: 20, r: 10, t: 40, b: 40 }
const x = scaleLinear().domain([-0.05, 1.05]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 13000]).range([H - m.b, m.t])
</script>

<template>
  <div class="mo">
    <div v-for="([k, t], i) in paneles" :key="k">
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <text :x="m.l" :y="22" class="titulo-eje" style="font-weight: 600">{{ t }}</text>
        <rect v-for="(c, j) in M.hist[k]" :key="j" :x="x(j / 10) - 13" :y="y(c)" width="26" :height="y(0) - y(c)" :fill="['#24398A', '#6B7385', '#D5433C'][i]" />
        <line :x1="x(0.5)" :x2="x(0.5)" :y1="m.t" :y2="H - m.b" stroke="#C9971C" stroke-width="2" stroke-dasharray="5 4" />
        <g class="eje"><line :x1="m.l" :x2="W - m.r" :y1="H - m.b" :y2="H - m.b" />
          <text v-for="t in [0, 0.5, 1]" :key="t" :x="x(t)" :y="H - m.b + 24" text-anchor="middle">{{ t }}</text></g>
      </svg>
    </div>
  </div>
</template>

<style scoped>.mo { display: grid; grid-template-columns: repeat(3, 1fr); gap: 28px; }</style>
