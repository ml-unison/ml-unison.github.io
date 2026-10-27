<script setup lang="ts">
// Qué días tienen registro cada presa: los huecos que isna() no ve.
import { scaleTime } from 'd3-scale'
import datos from '../datos/u01.json'

const P = (datos as any).presas
const nombres = Object.keys(P.cobertura).sort((a, b) => P.cobertura[a][0][0].localeCompare(P.cobertura[b][0][0]) || a.localeCompare(b))
const W = 1100, alto = 29, m = { l: 160, r: 30, t: 10 }
const H = m.t + nombres.length * alto + 40
const x = scaleTime().domain([new Date(P.inicio), new Date(P.fin)]).range([m.l, W - m.r])
const anios = ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01'].map(d => new Date(d))
const raro = (n: string) => P.cobertura[n].length > 1 || P.cobertura[n][0][0] !== P.inicio
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Días con registro por presa">
    <g class="rejilla"><line v-for="a in anios" :key="+a" :x1="x(a)" :x2="x(a)" :y1="m.t" :y2="H - 34" /></g>
    <g v-for="(n, i) in nombres" :key="n" :transform="`translate(0, ${m.t + i * alto})`">
      <text :x="m.l - 14" y="22" text-anchor="end" :style="{ fill: raro(n) ? '#1A2133' : '#5E6678', fontWeight: raro(n) ? 650 : 400 }">{{ n }}</text>
      <rect :x="m.l" y="8" :width="W - m.l - m.r" height="18" rx="3" fill="#FCEFEE" />
      <rect v-for="(t, k) in P.cobertura[n]" :key="k" :x="x(new Date(t[0]))" y="8" :width="Math.max(2, x(new Date(t[1])) - x(new Date(t[0])))" height="18" rx="3" fill="#24398A" />
    </g>
    <g class="eje"><text v-for="a in anios" :key="`t${+a}`" :x="x(a)" :y="H - 10" text-anchor="middle">{{ a.getUTCFullYear() }}</text></g>
  </svg>
</template>
