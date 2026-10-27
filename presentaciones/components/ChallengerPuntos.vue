<script setup lang="ts">
// Juntas tóricas con daño contra temperatura al despegue en los 23 vuelos anteriores al Challenger.
// Con «todos = false» solo se ven los vuelos con incidentes (lo que se discutió la noche anterior).
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u04.json'
const props = withDefaults(defineProps<{ todos?: boolean }>(), { todos: true })
const C = (datos as any).challenger
const W = 760, H = 380, m = { l: 64, r: 30, t: 40, b: 64 }
const x = scaleLinear().domain([28, 84]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 3]).range([H - m.b, m.t])
// apila los puntos repetidos (misma temperatura y número de juntas)
const pts = (() => { const vistos: Record<string, number> = {}
  return C.temp.map((t: number, i: number) => { const k = `${t}-${C.danadas[i]}`; vistos[k] = (vistos[k] ?? 0) + 1
    return { t, d: C.danadas[i], off: (vistos[k] - 1) * 12 } }) })()
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[30, 40, 50, 60, 70, 80]" :y-ticks="[0, 1, 2, 3]" x-titulo="temperatura al despegue (°F)" y-titulo="juntas con erosión o fuga (de 6)" />
    <rect :x="x(28)" :y="m.t" :width="x(53) - x(28)" :height="H - m.t - m.b" fill="#F6F7FA" />
    <line :x1="x(31)" :x2="x(31)" :y1="m.t" :y2="H - m.b" stroke="#D5433C" stroke-width="2.5" stroke-dasharray="6 5" />
    <text :x="x(31) + 8" :y="m.t + 18" style="fill: #1A2133; font-weight: 600">28 de enero de 1986: 31 °F</text>
    <text :x="x(33)" :y="m.t + 44" style="font-size: 16px">ningún vuelo previo</text>
    <text :x="x(33)" :y="m.t + 64" style="font-size: 16px">por debajo de 53 °F</text>
    <circle v-for="(p, i) in pts" v-show="todos || p.d > 0" :key="i" :cx="x(p.t)" :cy="y(p.d) - p.off" r="7" :fill="p.d > 0 ? '#D5433C' : '#24398A'" stroke="#fff" stroke-width="1.5" />
  </svg>
</template>
