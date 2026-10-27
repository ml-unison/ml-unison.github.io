<script setup lang="ts">
// Validación con origen móvil (130 orígenes, 2015–2025): MAE por horizonte (escala log) y cobertura de los intervalos del SARIMA.
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-02.json'
const M = (datos as any).movil
const props = withDefaults(defineProps<{ vista?: 'mae' | 'cobertura', ancho?: number, alto?: number }>(), { vista: 'mae', ancho: 640, alto: 420 })
const W = props.ancho, H = props.alto
const m = { l: 62, r: props.vista === 'mae' ? 200 : 80, t: 34, b: 60 }
const h = Array.from({ length: 12 }, (_, i) => i + 1)
const x = scaleLinear().domain([0.5, 12.5]).range([m.l, W - m.r])
const yl = scaleLog().domain([0.2, 15]).range([H - m.b, m.t])
const yc = scaleLinear().domain([0.6, 1]).range([H - m.b, m.t])
const series = [
  { k: 'lineal', n: 'lineal + mes', c: '#D5433C' },
  { k: 'ingenuo', n: 'estacional ingenuo', c: '#8A91A0' },
  { k: 'cuadratica', n: 'cuadrática + mes', c: '#C9971C' },
  { k: 'sarima', n: 'SARIMA', c: '#24398A' },
]
const camino = (v: number[], y: any) => line<number>().x((_, i) => x(h[i])).y(u => y(u))(v)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <template v-if="vista === 'mae'">
      <Ejes :x="x" :y="yl" :x-ticks="[1, 3, 6, 9, 12]" :y-ticks="[0.3, 1, 3, 10]" x-titulo="horizonte h (meses)" y-titulo="MAE (ppm, escala log)" />
      <g v-for="s in series" :key="s.k">
        <path :d="camino(M.mae[s.k], yl)" fill="none" :stroke="s.c" stroke-width="3.2" />
        <circle v-for="(v, i) in M.mae[s.k]" :key="i" :cx="x(h[i])" :cy="yl(v)" r="4.5" :fill="s.c" stroke="#fff" stroke-width="1.2" />
        <text :x="x(12) + 16" :y="yl(M.mae[s.k][11]) + 6" style="fill: #1A2133; font-weight: 600">{{ s.n }}</text>
      </g>
    </template>
    <template v-else>
      <Ejes :x="x" :y="yc" :x-ticks="[1, 3, 6, 9, 12]" :y-ticks="[0.6, 0.7, 0.8, 0.9, 1]" :y-formato="v => `${Math.round(100 * v)} %`" x-titulo="horizonte h (meses)" y-titulo="cobertura en prueba" />
      <line :x1="m.l" :x2="W - m.r" :y1="yc(0.8)" :y2="yc(0.8)" stroke="#24398A" stroke-dasharray="6 5" stroke-width="1.6" />
      <line :x1="m.l" :x2="W - m.r" :y1="yc(0.95)" :y2="yc(0.95)" stroke="#D5433C" stroke-dasharray="6 5" stroke-width="1.6" />
      <path :d="camino(M.cobertura80, yc)" fill="none" stroke="#24398A" stroke-width="3.2" />
      <path :d="camino(M.cobertura95, yc)" fill="none" stroke="#D5433C" stroke-width="3.2" />
      <circle v-for="(v, i) in M.cobertura80" :key="`a${i}`" :cx="x(h[i])" :cy="yc(v)" r="4.5" fill="#24398A" stroke="#fff" stroke-width="1.2" />
      <circle v-for="(v, i) in M.cobertura95" :key="`b${i}`" :cx="x(h[i])" :cy="yc(v)" r="4.5" fill="#D5433C" stroke="#fff" stroke-width="1.2" />
      <text :x="W - m.r + 12" :y="yc(0.95) + 6" style="fill: #1A2133; font-weight: 600">95 %</text>
      <text :x="W - m.r + 12" :y="yc(0.8) + 6" style="fill: #1A2133; font-weight: 600">80 %</text>
    </template>
  </svg>
</template>
