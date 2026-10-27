<script setup lang="ts">
// MAE en prueba contra el horizonte h para cada modelo. `mostrar` = cuántas series se ven (para revelar con clics).
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-03.json'
const props = withDefaults(defineProps<{ mostrar?: number }>(), { mostrar: 5 })
const D = (datos as any).horizonte
const series = [
  { k: 'ingenuo: semana pasada', n: 'ingenuo semanal', c: '#6B7385', d: true },
  { k: 'LightGBM directa', n: 'LightGBM directa', c: '#24398A', d: false },
  { k: 'LightGBM recursiva', n: 'LightGBM recursiva', c: '#D5433C', d: false },
  { k: 'red multisalida', n: 'red multisalida', c: '#C9971C', d: false },
  { k: 'lineal directa', n: 'lineal directa', c: '#8DA3C9', d: false },
]
const W = 780, H = 430, m = { l: 60, r: 190, t: 30, b: 60 }
const x = scaleLinear().domain([1, 24]).range([m.l, W - m.r])
const y = scaleLinear().domain([20, 80]).range([H - m.b, m.t])
const camino = (v: number[]) => line<number>().x((_, i) => x(i + 1)).y(t => y(t))(v)!
const vis = computed(() => series.slice(0, props.mostrar))
const etiquetas = computed(() => {
  const u = vis.value.map(s => ({ ...s, y: y(D[s.k][23]) })).sort((a, b) => a.y - b.y)
  for (let i = 1; i < u.length; i++) u[i].y = Math.max(u[i].y, u[i - 1].y + 23)
  return u
})
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[1, 4, 8, 12, 16, 20, 24]" :y-ticks="[20, 30, 40, 50, 60, 70, 80]" x-titulo="horizonte h (horas)" y-titulo="MAE en prueba (bicicletas por hora)" />
    <path v-for="s in vis" :key="s.k" :d="camino(D[s.k])" fill="none" :stroke="s.c" stroke-width="3.2" :stroke-dasharray="s.d ? '8 6' : undefined" />
    <g v-for="e in etiquetas" :key="`e${e.k}`">
      <line :x1="W - m.r + 8" :x2="W - m.r + 24" :y1="e.y" :y2="e.y" :stroke="e.c" stroke-width="3.2" />
      <text :x="W - m.r + 30" :y="e.y + 6" style="fill: #1A2133; font-weight: 600; font-size: 18px">{{ e.n }}</text>
    </g>
  </svg>
</template>
