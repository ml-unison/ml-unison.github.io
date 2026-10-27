<script setup lang="ts">
// EuroSAT: exactitud de prueba contra imágenes etiquetadas por clase, transferencia contra red desde cero.
import { scaleLog, scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-03.json'
const E = (datos as any).eurosat
const ks: number[] = E.k
const W = 600, H = 380, m = { l: 60, r: 220, t: 16, b: 70 }
const x = scaleLog().domain([ks[0] * 0.8, ks[ks.length - 1] * 1.25]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 100]).range([H - m.b, m.t])
const series = [
  { nombre: 'ResNet-18 de ImageNet', sub: '+ regresión logística', v: E.transfer, color: '#24398A' },
  { nombre: 'red pequeña desde cero', sub: 'mismas imágenes', v: E.cero, color: '#D5433C' },
]
const d = (v: number[]) => line<number>().x((_, i) => x(ks[i])).y(a => y(100 * a))(v)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="ks" :y-ticks="[0, 25, 50, 75, 100]" x-titulo="imágenes etiquetadas por clase" y-titulo="exactitud de prueba (%)" />
    <line :x1="x.range()[0]" :x2="x.range()[1]" :y1="y(10)" :y2="y(10)" stroke="#8A91A0" stroke-dasharray="3 4" />
    <text :x="x(ks[0])" :y="y(10) - 8">al azar: 10 %</text>
    <g v-for="s in series" :key="s.nombre">
      <path :d="d(s.v)" fill="none" :stroke="s.color" stroke-width="3.5" />
      <g v-for="(a, i) in s.v" :key="i">
        <circle :cx="x(ks[i])" :cy="y(100 * a)" r="6" :fill="s.color" />
        <text :x="x(ks[i])" :y="y(100 * a) + (s.color === '#24398A' ? -14 : 28)" text-anchor="middle" style="fill: #1A2133; font-size: 17px">{{ (100 * a).toFixed(1) }}</text>
      </g>
      <text :x="x(ks[ks.length - 1]) + 26" :y="y(100 * s.v[s.v.length - 1]) - 2" style="fill: #1A2133; font-weight: 600">{{ s.nombre }}</text>
      <text :x="x(ks[ks.length - 1]) + 26" :y="y(100 * s.v[s.v.length - 1]) + 20" style="font-size: 16px">{{ s.sub }}</text>
    </g>
  </svg>
</template>
