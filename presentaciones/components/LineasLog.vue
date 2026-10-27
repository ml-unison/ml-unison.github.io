<script setup lang="ts">
// Líneas con eje y (y opcionalmente x) logarítmico, etiquetas directas al final de cada serie.
import { computed } from 'vue'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import { SERIE } from './colores'

const props = withDefaults(defineProps<{
  x: number[] | Record<string, number[]>, series: Record<string, number[]>, xLog?: boolean, yLog?: boolean,
  xDominio?: [number, number], yDominio: [number, number], xTicks?: number[], yTicks?: number[],
  xTitulo?: string, yTitulo?: string, ancho?: number, alto?: number, colores?: string[], punteadas?: string[],
  marcadores?: boolean, formatoY?: (v: number) => string, formatoX?: (v: number) => string, derecha?: number,
}>(), { xLog: false, yLog: true, ancho: 760, alto: 420, marcadores: false, derecha: 150 })

const m = computed(() => ({ l: 70, r: props.derecha, t: 36, b: 60 }))
const xs = (n: string) => Array.isArray(props.x) ? props.x : props.x[n]
const todosX = computed(() => Array.isArray(props.x) ? props.x : Object.values(props.x).flat())
const x = computed(() => (props.xLog ? scaleLog() : scaleLinear()).domain(props.xDominio ?? [Math.min(...todosX.value), Math.max(...todosX.value)]).range([m.value.l, props.ancho - m.value.r]))
const y = computed(() => (props.yLog ? scaleLog() : scaleLinear()).domain(props.yDominio).range([props.alto - m.value.b, m.value.t]).clamp(true))
const nombres = computed(() => Object.keys(props.series))
const col = (k: number) => (props.colores ?? SERIE)[k % (props.colores ?? SERIE).length]
const camino = (n: string) => line<number>().defined(v => v != null && (!props.yLog || v > 0)).x((_, i) => x.value(xs(n)[i])).y(v => y.value(v))(props.series[n])!
const fy = (v: number) => props.formatoY ? props.formatoY(v) : (v >= 1000 ? v.toLocaleString('es-MX') : String(v))
const fx = (v: number) => props.formatoX ? props.formatoX(v) : (v >= 1000 ? v.toLocaleString('es-MX') : String(v))
const etiquetas = computed(() => {
  const u = nombres.value.map((n, k) => ({ n, k })).filter(o => o.n.trim() !== '').map(({ n, k }) => { const s = props.series[n]; let i = s.length - 1; while (i > 0 && (s[i] == null || (props.yLog && s[i] <= 0))) i--; return { n, k, y: y.value(s[i]) } })
  u.sort((a, b) => a.y - b.y)
  for (let i = 1; i < u.length; i++) u[i].y = Math.max(u[i].y, u[i - 1].y + 24)
  return u
})
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${alto}`" role="img">
    <g class="rejilla"><line v-for="t in (yTicks ?? [])" :key="t" :x1="m.l" :x2="ancho - m.r" :y1="y(t)" :y2="y(t)" /></g>
    <g class="eje">
      <line :x1="m.l" :x2="ancho - m.r" :y1="alto - m.b" :y2="alto - m.b" />
      <text v-for="t in (xTicks ?? x.ticks(5))" :key="`x${t}`" :x="x(t)" :y="alto - m.b + 26" text-anchor="middle">{{ fx(t) }}</text>
      <text v-for="t in (yTicks ?? [])" :key="`y${t}`" :x="m.l - 10" :y="y(t) + 6" text-anchor="end">{{ fy(t) }}</text>
      <text v-if="xTitulo" class="titulo-eje" :x="(m.l + ancho - m.r) / 2" :y="alto - 8" text-anchor="middle">{{ xTitulo }}</text>
      <text v-if="yTitulo" class="titulo-eje" :x="m.l - 60" :y="m.t - 16">{{ yTitulo }}</text>
    </g>
    <slot :x="x" :y="y" />
    <g v-for="(n, k) in nombres" :key="n">
      <path :d="camino(n)" fill="none" :stroke="col(k)" stroke-width="3.2" :stroke-dasharray="punteadas?.includes(n) ? '8 6' : undefined" stroke-linejoin="round" />
      <template v-if="marcadores"><circle v-for="(v, i) in series[n]" :key="i" v-show="v != null && (!yLog || v > 0)" :cx="x(xs(n)[i])" :cy="y(v)" r="5" :fill="col(k)" stroke="#fff" stroke-width="1.5" /></template>
    </g>
    <g v-for="e in etiquetas" :key="`e${e.n}`">
      <line :x1="ancho - m.r + 10" :x2="ancho - m.r + 26" :y1="e.y" :y2="e.y" :stroke="col(e.k)" stroke-width="3.2" :stroke-dasharray="punteadas?.includes(e.n) ? '5 4' : undefined" />
      <text :x="ancho - m.r + 32" :y="e.y + 6" style="fill: #1A2133; font-weight: 600">{{ e.n }}</text>
    </g>
  </svg>
</template>
