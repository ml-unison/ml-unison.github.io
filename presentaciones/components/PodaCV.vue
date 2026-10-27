<script setup lang="ts">
// Poda por costo–complejidad en las bicicletas: RECM de validación cruzada (por días) contra el número
// de hojas del subárbol podado, con banda de ± 1 error estándar.
import { scaleLinear, scaleLog } from 'd3-scale'
import { line, area } from 'd3-shape'
import datos from '../datos/u05-01.json'
const props = withDefaults(defineProps<{ ancho?: number, alto?: number }>(), { ancho: 640, alto: 430 })
const D = (datos as any).poda
// varias α consecutivas pueden dar el mismo árbol: se ordenan por hojas
const idx = (D.hojas as number[]).map((_, i) => i).sort((a, b) => D.hojas[a] - D.hojas[b])
const W = props.ancho, H = props.alto, m = { l: 64, r: 20, t: 34, b: 62 }
const x = scaleLog().domain([1.5, 1200]).range([m.l, W - m.r]), y = scaleLinear().domain([60, 150]).range([H - m.b, m.t])
const cam = line<number>().x(i => x(D.hojas[i])).y(i => y(D.cv[i]))(idx)!
const banda = area<number>().x(i => x(D.hojas[i])).y0(i => y(D.cv[i] - D.ee[i])).y1(i => y(D.cv[i] + D.ee[i]))(idx)!
const xi = D.hojas[D.i], yi = D.cv[D.i]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[2, 5, 10, 20, 50, 100, 200, 500, 1000]" :y-ticks="[60, 80, 100, 120, 140]" x-titulo="hojas del árbol podado (escala log)" y-titulo="RECM de validación cruzada" />
    <path :d="banda" fill="#24398A" opacity="0.15" />
    <path :d="cam" fill="none" stroke="#24398A" stroke-width="3" />
    <circle v-for="i in idx" :key="i" :cx="x(D.hojas[i])" :cy="y(D.cv[i])" r="3.5" fill="#24398A" />
    <circle :cx="x(xi)" :cy="y(yi)" r="11" fill="none" stroke="#D5433C" stroke-width="3" />
    <text :x="x(xi) + 16" :y="y(yi) - 34" style="fill: #1A2133; font-weight: 600">α = {{ D.alfa_cv }}: {{ D.hojas_cv }} hojas</text>
    <text :x="x(2)" :y="y(66)" style="font-size: 16px">← α grande: subajuste</text>
    <text :x="x(1100)" :y="y(96)" text-anchor="end" style="font-size: 16px">α chico: sobreajuste →</text>
  </svg>
</template>
