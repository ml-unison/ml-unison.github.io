<script setup lang="ts">
// Impureza de un nodo con dos clases (Gini, entropía/2, error) y, opcionalmente, la «cuerda» de la
// división A del ejemplo 80/20: padre p = 0.2, hijas p = 1/3 (60 datos) y p = 0 (40 datos).
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u05-01.json'
const props = withDefaults(defineProps<{ cuerda?: boolean, ancho?: number, alto?: number }>(), { cuerda: false, ancho: 580, alto: 420 })
const I = (datos as any).impureza
const W = props.ancho, H = props.alto, m = { l: 56, r: 110, t: 24, b: 66 }
const x = scaleLinear().domain([0, 1]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 0.52]).range([H - m.b, m.t])
const camino = (v: number[]) => line<number>().x((_, i) => x(I.p[i])).y(t => y(t))(v)!
const series = [
  { n: 'entropía / 2', v: I.entropia, c: '#C9971C' },
  { n: 'Gini', v: I.gini, c: '#24398A' },
  { n: 'error', v: I.error, c: '#D5433C' },
]
const g = (q: number) => 2 * q * (1 - q), e = (q: number) => Math.min(q, 1 - q)
const pI = 1 / 3, pD = 0, w = 0.6
const cuerdas = [
  { f: g, c: '#24398A' },
  { f: e, c: '#D5433C' },
].map(o => ({ ...o, a: [x(pD), y(o.f(pD))], b: [x(pI), y(o.f(pI))], prom: y(w * o.f(pI) + (1 - w) * o.f(pD)), padre: y(o.f(0.2)) }))
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 0.25, 0.5, 0.75, 1]" :y-ticks="[0, 0.1, 0.2, 0.3, 0.4, 0.5]" x-titulo="proporción p de la clase 2 en el nodo" y-titulo="impureza" />
    <path v-for="s in series" :key="s.n" :d="camino(s.v)" fill="none" :stroke="s.c" stroke-width="3.2" />
    <text v-for="(s, k) in series" :key="`t${s.n}`" :x="x(1) + 10" :y="[y(0.27), y(0.19), y(0.11)][k]" style="fill: #1A2133; font-weight: 600">{{ s.n }}</text>
    <template v-if="cuerda">
      <g v-for="(o, k) in cuerdas" :key="k">
        <line :x1="o.a[0]" :y1="o.a[1]" :x2="o.b[0]" :y2="o.b[1]" :stroke="o.c" stroke-width="2" stroke-dasharray="6 4" />
        <circle :cx="o.a[0]" :cy="o.a[1]" r="6" :fill="o.c" />
        <circle :cx="o.b[0]" :cy="o.b[1]" r="6" :fill="o.c" />
        <rect :x="x(0.2) - 7" :y="o.prom - 7" width="14" height="14" :fill="o.c" :transform="`rotate(45 ${x(0.2)} ${o.prom})`" />
        <circle :cx="x(0.2)" :cy="o.padre" r="8" fill="#fff" :stroke="o.c" stroke-width="3" />
      </g>
      <line :x1="x(0.2)" :x2="x(0.2)" :y1="y(0)" :y2="y(0.5)" stroke="#8A91A0" stroke-dasharray="2 4" />
      <text :x="x(0.2) + 4" :y="y(0.5) - 4" style="font-size: 16px">padre, p = 0.2</text>
      <text :x="x(0.35)" :y="y(0.29)" style="font-size: 16px">hija 1: p = 1/3</text>
      <text :x="x(0.09)" :y="y(0.02)" style="font-size: 16px">hija 2: p = 0</text>
    </template>
  </svg>
</template>
