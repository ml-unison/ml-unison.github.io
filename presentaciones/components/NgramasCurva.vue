<script setup lang="ts">
// Perplejidad de entrenamiento y validación de los n-gramas de caracteres del Quijote (n = 1…7).
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-04.json'
const G = (datos as any).ngramas
const props = withDefaults(defineProps<{ suave?: boolean }>(), { suave: true })
const W = 640, H = 420, m = { l: 56, r: 190, t: 34, b: 70 }
const x = scaleLinear().domain([1, 7]).range([m.l, W - m.r])
const y = scaleLog().domain([2.4, 22]).range([H - m.b, m.t])
const l = (v: number[]) => line<number>().x((_, i) => x(G.n[i])).y(p => y(p))(v)!
const series = [
  { k: 'a1', c: '#D5433C', alfa: 'α = 1' },
  { k: 'a001', c: '#24398A', alfa: 'α = 0.01' },
]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[1, 2, 3, 4, 5, 6, 7]" :y-ticks="[3, 4, 5, 7, 10, 15, 20]" x-titulo="n (contexto: las n − 1 letras anteriores)" y-titulo="perplejidad" />
    <g v-for="s in series.filter(s => props.suave || s.k === 'a1')" :key="s.k">
      <path :d="l(G[s.k].tr)" fill="none" :stroke="s.c" stroke-width="2.4" stroke-dasharray="7 5" opacity="0.7" />
      <path :d="l(G[s.k].va)" fill="none" :stroke="s.c" stroke-width="3.2" />
      <circle v-for="(v, i) in G[s.k].va" :key="i" :cx="x(G.n[i])" :cy="y(v)" r="4.5" :fill="s.c" />
      <text :x="x(7) + 12" :y="y(G[s.k].va[6]) + 6" class="etiqueta" style="fill: #1A2133">validación, {{ s.alfa }}</text>
      <text :x="x(7) + 12" :y="y(G[s.k].tr[6]) + 6" style="fill: #5E6678">entrenamiento</text>
    </g>
    <g v-if="props.suave">
      <circle :cx="x(5)" :cy="y(G.a001.va[4])" r="9" fill="none" stroke="#C9971C" stroke-width="3" />
      <text :x="x(5) + 4" :y="y(G.a001.va[4]) - 16" text-anchor="middle" style="fill: #1A2133; font-weight: 600">{{ G.a001.va[4].toFixed(2) }}</text>
    </g>
  </svg>
</template>
