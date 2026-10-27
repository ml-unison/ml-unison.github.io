<script setup lang="ts">
// Árbol de regresión de las bicicletas: RECM de entrenamiento y de prueba contra la profundidad máxima.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u05-01.json'
const props = withDefaults(defineProps<{ ancho?: number, alto?: number }>(), { ancho: 700, alto: 440 })
const D = (datos as any).profundidad
// la profundidad «sin límite» (31) se dibuja en una posición convencional
const pos = (p: number) => (p === 40 ? 24 : p)
const W = props.ancho, H = props.alto, m = { l: 64, r: 130, t: 34, b: 62 }
const x = scaleLinear().domain([1, 24.5]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 150]).range([H - m.b, m.t])
const camino = (v: number[]) => line<number>().x((_, i) => x(pos(D.prof[i]))).y(t => y(t))(v)!
const ticks = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24]
const fx = (v: number) => (v === 24 ? '∞' : String(v))
const iMin = D.prueba.indexOf(Math.min(...D.prueba))
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="ticks" :x-formato="fx" :y-ticks="[0, 25, 50, 75, 100, 125, 150]" x-titulo="profundidad máxima del árbol" y-titulo="RECM (bicicletas por hora)" />
    <line :x1="x(1)" :x2="x(24.5)" :y1="y(86.4)" :y2="y(86.4)" stroke="#8A91A0" stroke-dasharray="3 4" />
    <line :x1="x(1)" :x2="x(24.5)" :y1="y(72.2)" :y2="y(72.2)" stroke="#8A91A0" stroke-dasharray="3 4" />
    <text :x="x(1.3)" :y="y(86.4) - 6" style="font-size: 15px">M4 lineal: 86.4</text>
    <text :x="x(1.3)" :y="y(72.2) + 18" style="font-size: 15px">Poisson: 72.2</text>
    <path :d="camino(D.entrena)" fill="none" stroke="#6B7385" stroke-width="3" />
    <path :d="camino(D.prueba)" fill="none" stroke="#24398A" stroke-width="3.2" />
    <circle v-for="(p, i) in D.prof" :key="`e${i}`" :cx="x(pos(p))" :cy="y(D.entrena[i])" r="4.5" fill="#6B7385" />
    <circle v-for="(p, i) in D.prof" :key="`p${i}`" :cx="x(pos(p))" :cy="y(D.prueba[i])" r="4.5" fill="#24398A" />
    <circle :cx="x(D.prof[iMin])" :cy="y(D.prueba[iMin])" r="10" fill="none" stroke="#D5433C" stroke-width="2.5" />
    <text :x="x(D.prof[iMin]) - 6" :y="y(D.prueba[iMin]) + 32" text-anchor="middle" style="font-size: 16px">{{ D.prueba[iMin] }}</text>
    <text :x="x(24.5) + 8" :y="y(D.prueba[D.prueba.length - 1]) - 14" style="fill: #24398A; font-weight: 700">prueba</text>
    <text :x="x(24.5) + 8" :y="y(D.entrena[D.entrena.length - 1]) + 6" style="fill: #6B7385; font-weight: 700">entrenamiento</text>
  </svg>
</template>
