<script setup lang="ts">
// Adelie contra Gentoo (profundidad del pico y largo de la aleta, estandarizados): separables.
// «otras»: rectas de un perceptrón con distintas semillas; «margen»: la SVM de margen máximo.
import { scaleLinear } from 'd3-scale'
import { recta } from './recta'
import datos from '../datos/u04.json'
const props = withDefaults(defineProps<{ otras?: boolean, margen?: boolean, ancho?: number }>(), { otras: true, margen: true, ancho: 620 })
const D = (datos as any).svm_duro
const dom = [-2.6, 2.6, -2.3, 2.6]
const W = props.ancho, H = 440, m = { l: 20, r: 20, t: 20, b: 50 }
const x = scaleLinear().domain([dom[0], dom[1]]).range([m.l, W - m.r]), y = scaleLinear().domain([dom[2], dom[3]]).range([H - m.b, m.t])
const seg = (w: number[], b: number, c: number) => { const s = recta(w, b, c, dom); return s ? { x1: x(s[0][0]), y1: y(s[0][1]), x2: x(s[1][0]), y2: y(s[1][1]) } : null }
const esVS = new Set(D.vs as number[])
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" fill="#F6F7FA" rx="8" />
    <template v-if="otras"><line v-for="(o, k) in D.otras" :key="k" v-bind="seg(o.w, o.b, 0)!" stroke="#8A91A0" stroke-width="2.2" stroke-dasharray="7 5" /></template>
    <template v-if="margen">
      <line v-bind="seg(D.w, D.b, 1)!" stroke="#C9971C" stroke-width="2" stroke-dasharray="4 4" />
      <line v-bind="seg(D.w, D.b, -1)!" stroke="#C9971C" stroke-width="2" stroke-dasharray="4 4" />
      <line v-bind="seg(D.w, D.b, 0)!" stroke="#C9971C" stroke-width="3.5" />
    </template>
    <circle v-for="(v, i) in D.x" :key="i" :cx="x(v)" :cy="y(D.y[i])" r="5.5" :fill="D.g[i] ? '#D5433C' : '#24398A'" opacity="0.85" />
    <template v-if="margen"><circle v-for="i in D.vs" :key="`v${i}`" :cx="x(D.x[i])" :cy="y(D.y[i])" r="11" fill="none" stroke="#1A2133" stroke-width="2.2" /></template>
    <text :x="x(1.2)" :y="y(-1.9)" style="fill: #1A2133; font-weight: 700">Adelie</text>
    <text :x="x(-2.4)" :y="y(2.3)" style="fill: #1A2133; font-weight: 700">Gentoo</text>
    <text :x="W / 2" :y="H - 16" text-anchor="middle" class="titulo-eje">profundidad del pico → · largo de la aleta ↑ (estandarizados)</text>
  </svg>
</template>
