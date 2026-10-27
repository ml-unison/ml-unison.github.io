<script setup lang="ts">
// Temperatura media diaria en el aeropuerto de Hermosillo, 1992–2024 (o un año de cerca, con su climatología).
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-01.json'
const props = withDefaults(defineProps<{ zoom?: boolean, ancho?: number, alto?: number }>(), { zoom: false, ancho: 1120, alto: 400 })
const D = (datos as any)
const m = { l: 56, r: 130, t: 30, b: 50 }
const dia0 = Date.UTC(1992, 0, 1)
const iPrueba = (Date.UTC(2021, 0, 1) - dia0) / 864e5
const y = computed(() => scaleLinear().domain([5, 43]).range([props.alto - m.b, m.t]))
const x = computed(() => props.zoom
  ? scaleLinear().domain([1, 366]).range([m.l, props.ancho - m.r])
  : scaleLinear().domain([1992, 2025]).range([m.l, props.ancho - m.r]))
const anio = (i: number) => 1992 + i / 365.2425
const camino = computed(() => props.zoom
  ? line<number>().x((_, i) => x.value(D.anio2023.dia[i])).y(v => y.value(v))(D.anio2023.y)!
  : line<number>().x((_, i) => x.value(anio(i))).y(v => y.value(v))(D.serie.y)!)
const clim = computed(() => line<number>().x((_, i) => x.value(D.anio2023.dia[i])).y(v => y.value(v))(D.anio2023.clim)!)
const meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const inicioMes = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${alto}`" role="img">
    <Ejes v-if="!zoom" :x="x" :y="y" :x-ticks="[1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]" :x-formato="v => String(v)" :y-ticks="[10, 20, 30, 40]" y-titulo="°C" />
    <Ejes v-else :x="x" :y="y" :x-ticks="inicioMes" :x-formato="v => meses[inicioMes.indexOf(v)]" :y-ticks="[10, 20, 30, 40]" y-titulo="°C · 2023" />
    <template v-if="!zoom">
      <rect :x="x(anio(iPrueba))" :y="m.t" :width="x(2025) - x(anio(iPrueba))" :height="alto - m.t - m.b" fill="#C0CEE3" opacity="0.55" />
      <text :x="x(2023)" :y="alto - m.b - 12" text-anchor="middle" style="fill: #1A2133; font-weight: 600">prueba</text>
      <path :d="camino" fill="none" stroke="#24398A" stroke-width="0.6" />
    </template>
    <template v-else>
      <path :d="camino" fill="none" stroke="#1A2133" stroke-width="2" stroke-linejoin="round" />
      <path :d="clim" fill="none" stroke="#24398A" stroke-width="5" opacity="0.85" />
      <text :x="ancho - m.r + 10" :y="y(D.anio2023.clim.at(-1)) + 6" style="fill: #1A2133; font-weight: 600">climatología</text>
      <text :x="ancho - m.r + 10" :y="y(D.anio2023.y.at(-1)) + 30" style="fill: #1A2133; font-weight: 600">2023</text>
    </template>
  </svg>
</template>
