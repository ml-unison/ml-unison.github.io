<script setup lang="ts">
// Nueve grupos en rejilla, k = 9: el peor y el mejor resultado de 60 inicios al azar,
// y la inercia final de cada inicio (al azar contra k-medias++).
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u02.json'
const R = (datos as any).rejilla, C = (datos as any).clientes
const nueve = ['#24398A', '#D5433C', '#C9971C', '#6B7385', '#496EA5', '#946721', '#20305A', '#E58E86', '#8DA3C9']
const W = 400, H = 250, m = { l: 120, r: 16, t: 20, b: 56 }
const x = scaleLinear().domain([0, 70]).range([m.l, W - m.r])
const fila = { azar: 60, pp: 150 }
const jit = (i: number) => ((i * 37) % 23 - 11) * 2.2
const mal = (C.inits_azar as number[]).filter(v => v > 13.2).length
const malpp = (C.inits_pp as number[]).filter(v => v > 13.2).length
</script>

<template>
  <div class="ml">
    <Dispersion :x="R.x" :y="R.y" :g="R.e_peor" :colores="nueve" :centros="R.C_peor" :ancho="300" :alto="300" :radio="3.6" titulo="El peor inicio" />
    <Dispersion :x="R.x" :y="R.y" :g="R.e_mejor" :colores="nueve" :centros="R.C_mejor" :ancho="300" :alto="300" :radio="3.6" titulo="El mejor inicio" />
    <div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <g class="eje"><line :x1="m.l" :x2="W - m.r" :y1="H - m.b" :y2="H - m.b" />
          <text v-for="t in [0, 20, 40, 60]" :key="t" :x="x(t)" :y="H - m.b + 26" text-anchor="middle">{{ t }}</text>
          <text class="titulo-eje" :x="(m.l + W - m.r) / 2" :y="H - 4" text-anchor="middle">J final</text>
          <text :x="m.l - 14" :y="fila.azar + 6" text-anchor="end" style="fill: #1A2133">al azar</text>
          <text :x="m.l - 14" :y="fila.pp + 6" text-anchor="end" style="fill: #1A2133">k-medias++</text>
        </g>
        <circle v-for="(v, i) in C.inits_azar" :key="`a${i}`" :cx="x(v)" :cy="fila.azar + jit(i)" r="6" fill="#D5433C" opacity="0.6" />
        <circle v-for="(v, i) in C.inits_pp" :key="`p${i}`" :cx="x(v)" :cy="fila.pp + jit(i)" r="6" fill="#24398A" opacity="0.6" />
      </svg>
      <p class="ml__nota">60 inicios de cada tipo. Quedan atrapados en un mínimo local: <b>{{ mal }}</b> de 60 al azar, <b>{{ malpp }}</b> de 60 con k-medias++.</p>
    </div>
  </div>
</template>

<style scoped>
.ml { display: grid; grid-template-columns: 300px 300px 1fr; gap: 24px; align-items: center; }
.ml__nota { font-size: 19px; color: var(--tenue); margin-top: 8px; }
.ml__nota b { color: var(--tinta); }
</style>
