<script setup lang="ts">
// Norma del gradiente de los pesos de cada capa (relativa a la última) en una red de 40 capas recién inicializada.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-03.json'
const R = (datos as any).residual as Record<string, number[]>
const L = R.residual.length
const W = 620, H = 360, m = { l: 70, r: 16, t: 16, b: 70 }
const x = scaleLinear().domain([0, L + 1]).range([m.l, W - m.r])
const y = scaleLinear().domain([-18, 3]).range([H - m.b, m.t])
const series = [
  { k: 'residual', color: '#24398A', et: 'residual', dx: 2, dy: -2.2 },
  { k: 'simple, inicialización de He', color: '#C9971C', et: 'simple, inicialización de He', dx: 2, dy: 1.6 },
  { k: 'simple', color: '#D5433C', et: 'simple', dx: 13, dy: 1.6 },
]
const d = (v: number[]) => line<number>().x((_, i) => x(i + 1)).y(a => y(a))(v)!
const pot = (t: number) => (t === 0 ? '1' : `10${String(t).replace('-', '⁻').replace(/\d/g, c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c])}`)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[1, 10, 20, 30, 40]" :y-ticks="[-18, -12, -6, 0]" :y-formato="pot"
          x-titulo="capa (1 = junto a la entrada)" y-titulo="norma del gradiente (última capa = 1)" />
    <g v-for="s in series" :key="s.k">
      <path :d="d(R[s.k])" fill="none" :stroke="s.color" stroke-width="3" />
      <text :x="x(s.dx)" :y="y(R[s.k][s.dx - 1] + s.dy)" style="fill: #1A2133; font-weight: 600">{{ s.et }}</text>
    </g>
  </svg>
</template>
