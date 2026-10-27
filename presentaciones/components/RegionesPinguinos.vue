<script setup lang="ts">
// Pingüinos con las dos medidas del pico: regiones de decisión y elipses de covarianza (2 desviaciones)
// de LDA (covarianza común) o QDA (una por especie).
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import { SERIE } from './colores'
import datos from '../datos/u04.json'
const props = withDefaults(defineProps<{ tipo?: 'lda' | 'qda', ancho?: number }>(), { tipo: 'lda', ancho: 600 })
const P = (datos as any).pinguinos
const W = props.ancho, H = 430, m = { l: 64, r: 16, t: 30, b: 64 }
const x = scaleLinear().domain([31, 61]).range([m.l, W - m.r]), y = scaleLinear().domain([12.5, 22]).range([H - m.b, m.t])
const l = line<number[]>().x(p => x(p[0])).y(p => y(p[1]))
const elipses = (props.tipo === 'lda' ? P.elipses_lda : P.elipses_qda).map((e: number[][]) => l([...e, e[0]]))
const nombres = [{ t: 'Adelie', x: 35, y: 21.3 }, { t: 'Chinstrap', x: 51.5, y: 21.3 }, { t: 'Gentoo', x: 53, y: 13.3 }]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <clipPath :id="`rp${tipo}`"><rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" /></clipPath>
    <g :clip-path="`url(#rp${tipo})`"><Regiones :datos="P[tipo]" :x="x" :y="y" :opacidad="0.55" /></g>
    <Ejes :x="x" :y="y" :x-ticks="[35, 40, 45, 50, 55, 60]" :y-ticks="[14, 16, 18, 20, 22]" x-titulo="largo del pico (mm)" y-titulo="profundidad del pico (mm)" />
    <circle v-for="(v, i) in P.x" :key="i" :cx="x(v)" :cy="y(P.y[i])" r="4" :fill="SERIE[P.g[i]]" stroke="#fff" stroke-width="0.8" />
    <path v-for="(e, k) in elipses" :key="`e${k}`" :d="e" fill="none" :stroke="SERIE[k]" stroke-width="2.8" />
    <text v-for="(n, k) in nombres" :key="n.t" :x="x(n.x)" :y="y(n.y)" style="fill: #1A2133; font-weight: 700">{{ n.t }}</text>
  </svg>
</template>
