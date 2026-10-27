<script setup lang="ts">
// Descomposición STL de la temperatura diaria de Hermosillo: serie = tendencia + estacionalidad + residuo.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-01.json'
const S = (datos as any).stl
const W = 1120, alto = 92, sep = 10, m = { l: 190, r: 20 }
const anio = (i: number) => 1992 + (i * S.paso) / 365.2425
const x = scaleLinear().domain([1992, 2025]).range([m.l, W - m.r])
const paneles = [
  { nombre: 'serie', d: S.serie, dom: [5, 43], ticks: [10, 40], color: '#24398A', ancho: 0.7 },
  { nombre: 'tendencia', d: S.tendencia, dom: [24, 28.5], ticks: [24, 28], color: '#D5433C', ancho: 3 },
  { nombre: 'estacionalidad', d: S.estacional, dom: [-15, 14], ticks: [-10, 10], color: '#C9971C', ancho: 0.9 },
  { nombre: 'residuo', d: S.residuo, dom: [-9, 9], ticks: [-5, 5], color: '#6B7385', ancho: 0.6 },
].map((p, k) => {
  const y0 = k * (alto + sep)
  const y = scaleLinear().domain(p.dom).range([y0 + alto, y0 + 6])
  return { ...p, y, y0, camino: line<number>().x((_, i) => x(anio(i))).y(v => y(v))(p.d)! }
})
const H = 4 * (alto + sep) + 34
const anios = [1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <g v-for="p in paneles" :key="p.nombre">
      <line v-for="t in p.ticks" :key="t" :x1="m.l" :x2="W - m.r" :y1="p.y(t)" :y2="p.y(t)" stroke="#E2E5EC" />
      <text v-for="t in p.ticks" :key="`t${t}`" :x="m.l - 10" :y="p.y(t) + 6" text-anchor="end">{{ String(t).replace('-', '−') }}</text>
      <text :x="0" :y="p.y0 + alto / 2 + 7" style="fill: #1A2133; font-weight: 600; font-size: 21px">{{ p.nombre }}</text>
      <path :d="p.camino" fill="none" :stroke="p.color" :stroke-width="p.ancho" />
    </g>
    <text v-for="a in anios" :key="a" :x="x(a)" :y="H - 6" text-anchor="middle">{{ a }}</text>
  </svg>
</template>
