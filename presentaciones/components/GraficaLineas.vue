<script setup lang="ts">
// Gráfica de líneas con etiquetas directas (sin leyenda) y línea de referencia opcional.
import { computed } from 'vue'
import { scaleLinear, scaleTime } from 'd3-scale'
import { line } from 'd3-shape'
import { SERIE } from './colores'

const props = withDefaults(defineProps<{
  x: (string | number)[], series: Record<string, number[]>, fechas?: boolean,
  yDominio?: [number, number], yTicks?: number[], xTicks?: (string | number)[], xEtiquetas?: string[],
  yTitulo?: string, referencia?: { y: number, texto: string, x?: number }, ancho?: number, alto?: number, marcadores?: boolean, sinEtiquetas?: boolean,
}>(), { sinEtiquetas: false, fechas: false, ancho: 1000, alto: 400, marcadores: false })

const m = { l: 56, r: props.sinEtiquetas ? 20 : 150, t: 34, b: 40 }
const xs = computed(() => props.x.map(v => props.fechas ? new Date(v as string) : (v as number)))
const x = computed(() => {
  const d = xs.value
  const s = props.fechas ? scaleTime() : scaleLinear()
  return (s as any).domain([d[0], d[d.length - 1]]).range([m.l, props.ancho - m.r])
})
const todos = computed(() => Object.values(props.series).flat())
const y = computed(() => scaleLinear().domain(props.yDominio ?? [0, Math.max(...todos.value) * 1.1]).range([props.alto - m.b, m.t]))
const nombres = computed(() => Object.keys(props.series))
const camino = (vals: number[]) => line<number>().defined(v => v != null).x((_, i) => x.value(xs.value[i])).y(v => y.value(v))(vals)!
const formatoX = (v: any) => props.fechas ? String(new Date(v).getUTCFullYear()) : String(v)
const ticksX = computed(() => props.xTicks ? props.xTicks.map(v => props.fechas ? new Date(v as string) : v) : x.value.ticks(6))
// Etiquetas al final de cada serie, repartidas para no encimarse
const etiquetas = computed(() => {
  const ult = nombres.value.map((n, k) => ({ n, k, y: y.value(props.series[n][props.series[n].length - 1]) }))
  ult.sort((a, b) => a.y - b.y)
  for (let i = 1; i < ult.length; i++) ult[i].y = Math.max(ult[i].y, ult[i - 1].y + 24)
  return ult
})
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${alto}`" role="img">
    <g class="rejilla"><line v-for="t in (yTicks ?? y.ticks(5))" :key="t" :x1="m.l" :x2="ancho - m.r" :y1="y(t)" :y2="y(t)" /></g>
    <g class="eje">
      <line :x1="m.l" :x2="ancho - m.r" :y1="alto - m.b" :y2="alto - m.b" />
      <text v-for="(t, i) in ticksX" :key="`x${i}`" :x="x(t)" :y="alto - m.b + 26" text-anchor="middle">{{ xEtiquetas ? xEtiquetas[i] : formatoX(t) }}</text>
      <text v-for="t in (yTicks ?? y.ticks(5))" :key="`y${t}`" :x="m.l - 10" :y="y(t) + 6" text-anchor="end">{{ t }}</text>
      <text v-if="yTitulo" class="titulo-eje" :x="m.l" :y="m.t - 14">{{ yTitulo }}</text>
    </g>
    <template v-if="referencia">
      <line :x1="m.l" :x2="ancho - m.r" :y1="y(referencia.y)" :y2="y(referencia.y)" stroke="#8A91A0" stroke-width="1.5" stroke-dasharray="6 6" />
      <text :x="m.l + (ancho - m.l - m.r) * (referencia.x ?? 0.3)" :y="y(referencia.y) - 8" style="font-size: 15px">{{ referencia.texto }}</text>
    </template>
    <g v-for="(n, k) in nombres" :key="n">
      <path :d="camino(series[n])" fill="none" :stroke="SERIE[k]" stroke-width="3.2" stroke-linejoin="round" />
      <template v-if="marcadores">
        <circle v-for="(v, i) in series[n]" :key="i" :cx="x(xs[i])" :cy="y(v)" r="5" :fill="SERIE[k]" stroke="#fff" stroke-width="1.5" />
      </template>
    </g>
    <g v-for="e in (sinEtiquetas ? [] : etiquetas)" :key="`e${e.n}`">
      <line :x1="ancho - m.r + 10" :x2="ancho - m.r + 26" :y1="e.y" :y2="e.y" :stroke="SERIE[e.k]" stroke-width="3.2" />
      <text :x="ancho - m.r + 32" :y="e.y + 6" style="fill: #1A2133; font-weight: 600">{{ e.n }}</text>
    </g>
  </svg>
</template>
