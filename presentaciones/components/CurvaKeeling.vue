<script setup lang="ts">
// La curva de Keeling (CO₂ mensual en Mauna Loa) completa y un acercamiento a 2011–2014 que muestra el ciclo anual.
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-02.json'
const props = withDefaults(defineProps<{ prueba?: boolean, zoom?: boolean }>(), { prueba: true, zoom: true })
const K = (datos as any).keeling
const n = K.n_ent
const W = 1128, H = 400
const izq = computed(() => (props.zoom ? 720 : W))
const m = { l: 62, r: 20, t: 34, b: 56 }
const x = computed(() => scaleLinear().domain([1957, 2028]).range([m.l, izq.value - m.r]))
const y = scaleLinear().domain([300, 436]).range([H - m.b, m.t])
const tr = computed(() => line<number>().x((_, i) => x.value(K.t[i])).y(v => y(v))(K.y.slice(0, n))!)
const te = computed(() => line<number>().x((_, i) => x.value(K.t[n + i])).y(v => y(v))(K.y.slice(n))!)
// acercamiento: enero de 2011 a diciembre de 2014
const i0 = K.t.findIndex((t: number) => t > 2011), i1 = n
const zx = scaleLinear().domain([2011, 2015]).range([izq.value + 66, W - 10])
const zy = scaleLinear().domain([388, 404]).range([H - m.b, m.t])
const zc = computed(() => line<number>().x((_, i) => zx(K.t[i0 + i])).y(v => zy(v))(K.y.slice(i0, i1))!)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <g class="rejilla"><line v-for="t in [320, 340, 360, 380, 400, 420]" :key="t" :x1="m.l" :x2="izq - m.r" :y1="y(t)" :y2="y(t)" /></g>
    <g class="eje">
      <line :x1="m.l" :x2="izq - m.r" :y1="H - m.b" :y2="H - m.b" />
      <text v-for="t in [1960, 1970, 1980, 1990, 2000, 2010, 2020]" :key="t" :x="x(t)" :y="H - m.b + 26" text-anchor="middle">{{ t }}</text>
      <text v-for="t in [320, 340, 360, 380, 400, 420]" :key="`y${t}`" :x="m.l - 10" :y="y(t) + 6" text-anchor="end">{{ t }}</text>
      <text class="titulo-eje" :x="m.l - 50" :y="m.t - 14">CO₂ (ppm)</text>
    </g>
    <path :d="tr" fill="none" stroke="#24398A" stroke-width="2" />
    <template v-if="prueba">
      <path :d="te" fill="none" stroke="#8A91A0" stroke-width="2" />
      <line :x1="x(2015)" :x2="x(2015)" :y1="m.t" :y2="H - m.b" stroke="#5E6678" stroke-dasharray="5 5" />
      <text :x="x(2014) " :y="y(424)" text-anchor="end" style="fill: #24398A; font-weight: 600">entrenamiento</text>
      <text :x="x(2016)" :y="y(392)" style="fill: #5E6678; font-weight: 600">prueba</text>
    </template>
    <circle :cx="x(K.t[0])" :cy="y(K.y[0])" r="6" fill="#D5433C" />
    <text :x="x(K.t[0]) + 4" :y="y(K.y[0]) + 30" style="fill: #1A2133; font-weight: 600">marzo de 1958: {{ K.primero.toFixed(1) }}</text>
    <template v-if="zoom">
      <rect :x="x(2011)" :y="y(404)" :width="x(2015) - x(2011)" :height="y(388) - y(404)" fill="none" stroke="#C9971C" stroke-width="2" />
      <g class="rejilla"><line v-for="t in [390, 395, 400]" :key="t" :x1="zx(2011)" :x2="zx(2015)" :y1="zy(t)" :y2="zy(t)" /></g>
      <rect :x="zx(2011) - 8" :y="m.t - 4" :width="zx(2015) - zx(2011) + 16" :height="H - m.b - m.t + 8" fill="none" stroke="#C9971C" stroke-width="2" rx="6" />
      <g class="eje">
        <text v-for="t in [2011, 2012, 2013, 2014, 2015]" :key="`z${t}`" :x="zx(t)" :y="H - m.b + 26" text-anchor="middle">{{ t }}</text>
        <text v-for="t in [390, 395, 400]" :key="`zy${t}`" :x="zx(2011) - 12" :y="zy(t) + 6" text-anchor="end">{{ t }}</text>
      </g>
      <path :d="zc" fill="none" stroke="#24398A" stroke-width="2.4" />
      <circle v-for="(v, i) in K.y.slice(i0, i1)" :key="i" :cx="zx(K.t[i0 + i])" :cy="zy(v)" r="3.6" fill="#24398A" />
      <text :x="zx(2011) + 6" :y="m.t + 24" style="fill: #1A2133; font-weight: 600">2011–2014: el ciclo anual</text>
    </template>
  </svg>
</template>
