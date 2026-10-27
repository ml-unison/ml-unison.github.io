<script setup lang="ts">
// Desviación estándar del error de pronóstico contra el horizonte: teoría (líneas) y 20,000 simulaciones (puntos).
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-02.json'
const V = (datos as any).horizonte
const props = withDefaults(defineProps<{ ancho?: number, alto?: number }>(), { ancho: 640, alto: 420 })
const W = props.ancho, H = props.alto, m = { l: 56, r: 190, t: 40, b: 60 }
const h = Array.from({ length: 24 }, (_, i) => i + 1)
const x = scaleLinear().domain([0, 24]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 5]).range([H - m.b, m.t])
const series = [
  { k: 'caminata', n: 'caminata: √h', c: '#D5433C' },
  { k: 'ar09', n: 'AR(1), φ = 0.9', c: '#24398A' },
  { k: 'ar05', n: 'AR(1), φ = 0.5', c: '#C9971C' },
]
const camino = (k: string) => line<number>().x((_, i) => x(h[i])).y(v => y(v))(V[k].teoria)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[1, 6, 12, 18, 24]" :y-ticks="[0, 1, 2, 3, 4, 5]" x-titulo="horizonte h" y-titulo="desv. est. del error (σ = 1)" />
    <g v-for="s in series" :key="s.k">
      <path :d="camino(s.k)" fill="none" :stroke="s.c" stroke-width="3.2" />
      <circle v-for="(v, i) in V[s.k].simulacion" v-show="i % 2 === 0" :key="i" :cx="x(h[i])" :cy="y(v)" r="5.5" :fill="s.c" stroke="#fff" stroke-width="1.5" />
      <text :x="x(24) + 12" :y="y(V[s.k].teoria[23]) + 6" style="fill: #1A2133; font-weight: 600">{{ s.n }}</text>
    </g>
  </svg>
</template>
