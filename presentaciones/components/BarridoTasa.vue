<script setup lang="ts">
// SGD con cinco tasas de aprendizaje: pérdida del minilote (promedio móvil de 10 pasos) en tres épocas.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-02.json'
const B = (datos as any).barrido as { eta: number, perdida: number[], diverge: boolean, final?: number }[]
const W = 760, H = 420, m = { l: 64, r: 130, t: 40, b: 70 }
const x = scaleLinear().domain([0, 237]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 2.8]).range([H - m.b, m.t]).clamp(true)
const estilo: Record<string, { c: string, d?: string }> = {
  '0.001': { c: '#6B7385' }, '0.01': { c: '#C9971C' }, '0.1': { c: '#24398A' }, '0.5': { c: '#24398A', d: '3 5' }, '1': { c: '#D5433C' },
}
const curvas = B.map(b => ({ ...b, s: estilo[String(b.eta)], d: line<number>().x((_, i) => x(i)).y(v => y(v))(b.perdida)! }))
// etiquetas a la derecha sin encimarse
const et = curvas.filter(c => !c.diverge).map(c => ({ eta: c.eta, s: c.s, y: y(c.perdida[c.perdida.length - 1]) })).sort((a, b) => a.y - b.y)
for (let i = 1; i < et.length; i++) et[i].y = Math.max(et[i].y, et[i - 1].y + 24)
const fmt = (v: number) => String(v)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 79, 158, 237]" :y-ticks="[0, 0.5, 1, 1.5, 2, 2.5]" x-titulo="paso (minilote de 128); tres épocas" y-titulo="pérdida del minilote" />
    <path v-for="c in curvas" :key="c.eta" :d="c.d" fill="none" :stroke="c.s.c" stroke-width="3" :stroke-dasharray="c.s.d" stroke-linejoin="round" />
    <text :x="x(12)" :y="y(2.62)" style="fill: #D5433C; font-weight: 600">η = 1: diverge</text>
    <g v-for="e in et" :key="`e${e.eta}`">
      <line :x1="W - m.r + 10" :x2="W - m.r + 30" :y1="e.y" :y2="e.y" :stroke="e.s.c" stroke-width="3" :stroke-dasharray="e.s.d" />
      <text :x="W - m.r + 36" :y="e.y + 6" style="fill: #1A2133; font-weight: 600">η = {{ fmt(e.eta) }}</text>
    </g>
  </svg>
</template>
