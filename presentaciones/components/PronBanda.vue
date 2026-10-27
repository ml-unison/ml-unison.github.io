<script setup lang="ts">
// Una semana de prueba pronosticada con 24 horas de anticipación: pronóstico, intervalo del 10 % al 90 % y lo observado.
import { scaleLinear } from 'd3-scale'
import { line, area } from 'd3-shape'
import datos from '../datos/u07-03.json'
const props = withDefaults(defineProps<{ cual?: 'noviembre' | 'sandy', banda?: boolean }>(), { cual: 'noviembre', banda: true })
const S = (datos as any)[props.cual]
const W = 1120, H = 360, m = { l: 66, r: 20, t: 40, b: 40 }
const n = S.real.length
const x = scaleLinear().domain([0, n - 1]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 850]).range([H - m.b, m.t])
const DIA = ['dom', 'lun', 'mar', 'mié', 'jue', 'vie', 'sáb']
const i0 = new Date(S.inicio + 'T00:00:00')
const dias = Array.from({ length: n / 24 }, (_, k) => { const f = new Date(i0.getTime() + k * 864e5); return { k, t: `${DIA[f.getDay()]} ${f.getDate()}` } })
const banda = area<number>().x((_, i) => x(i)).y0((_, i) => y(S.q10[i])).y1(v => y(v))(S.q90)!
const linea = (v: number[]) => line<number>().x((_, i) => x(i)).y(t => y(t))(v)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <rect v-if="cual === 'sandy'" :x="x(72)" :y="m.t" :width="x(120) - x(72)" :height="H - m.t - m.b" fill="#FCEFEE" />
    <text v-if="cual === 'sandy'" :x="x(96)" :y="m.t - 10" text-anchor="middle" style="fill: #B23028; font-weight: 700">huracán Sandy</text>
    <Ejes :x="x" :y="y" :x-ticks="[]" :y-ticks="[0, 200, 400, 600, 800]" y-titulo="rentas por hora" />
    <line v-for="d in dias" :key="`l${d.k}`" :x1="x(d.k * 24)" :x2="x(d.k * 24)" :y1="m.t" :y2="H - m.b" stroke="#E2E5EC" />
    <text v-for="d in dias" :key="d.k" :x="x(d.k * 24 + 12)" :y="H - m.b + 26" text-anchor="middle">{{ d.t }}</text>
    <path v-if="banda" :d="banda" fill="#C0CEE3" opacity="0.85" />
    <path :d="linea(S.pred)" fill="none" stroke="#24398A" stroke-width="2.6" />
    <path :d="linea(S.real)" fill="none" stroke="#1A2133" stroke-width="1.8" />
  </svg>
</template>
