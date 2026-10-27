<script setup lang="ts">
// Fracción de la ganancia de LightGBM por atributo, a 1 hora y a 24 horas.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u07-03.json'
const I = (datos as any).importancia
const W = 520, H = 300, m = { l: 150, r: 50, t: 40, b: 20 }
const x = scaleLinear().domain([0, 0.7]).range([m.l, W - m.r])
const paneles = [{ k: 'h1', t: 'a 1 hora (h = 1)' }, { k: 'h24', t: 'a 24 horas (h = 24)' }]
const fila = 40
</script>

<template>
  <div class="imp">
    <svg v-for="p in paneles" :key="p.k" class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <text :x="m.l" :y="22" style="fill: #20305A; font-weight: 700; font-size: 20px">{{ p.t }}</text>
      <g v-for="(a, i) in I[p.k]" :key="a.nombre">
        <text :x="m.l - 10" :y="m.t + i * fila + 26" text-anchor="end" style="fill: #1A2133">{{ a.nombre }}</text>
        <rect :x="m.l" :y="m.t + i * fila + 8" :width="x(a.valor) - m.l" height="26" :fill="a.nombre.startsWith('y(') ? '#24398A' : '#C9971C'" />
        <text :x="x(a.valor) + 8" :y="m.t + i * fila + 27" style="font-size: 17px">{{ Math.round(100 * a.valor) }} %</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.imp { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
</style>
