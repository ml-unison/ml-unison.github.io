<script setup lang="ts">
// Los 3! = 6 órdenes de llegada de los atributos del modelo de juguete: contribuciones marginales apiladas (miles de USD).
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u05-04.json'
const J = (datos as any).juguete
const props = withDefaults(defineProps<{ promedio?: boolean }>(), { promedio: true })
const COL: Record<string, string> = { area: '#24398A', calidad: '#D5433C', anio: '#C9971C' }
const NOM: Record<string, string> = { area: 'área', calidad: 'calidad', anio: 'año' }
const v0 = J.v.find((e: any) => e[0].length === 0)[1], v1 = J.v.find((e: any) => e[0].length === 3)[1]
const W = 1100, fila = 50, m = { l: 250, r: 30, t: 30, b: 56 }
const x = scaleLinear().domain([170, 330]).range([m.l, W - m.r])
const filas = [
  ...J.ordenes.map((o: any) => ({ t: o.orden.map((k: string) => NOM[k]).join(' → '), seg: o.orden.map((k: string) => [k, o.contrib[k]]) })),
  ...(props.promedio ? [{ t: 'promedio = Shapley', seg: ['area', 'calidad', 'anio'].map(k => [k, J.shapley[k]]), shapley: true }] : []),
]
const tramos = filas.map((f: any) => { let a = v0; return f.seg.map(([k, c]: [string, number]) => { const s = { k, c, x0: x(a), x1: x(a + c) }; a += c; return s }) })
const H = m.t + 7 * fila + m.b
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <g class="rejilla"><line v-for="t in [180, 200, 220, 240, 260, 280, 300, 320]" :key="t" :x1="x(t)" :x2="x(t)" :y1="m.t - 8" :y2="m.t + 7 * fila" /></g>
    <line :x1="x(v0)" :x2="x(v0)" :y1="m.t - 14" :y2="m.t + 7 * fila" stroke="#5E6678" stroke-dasharray="5 4" stroke-width="1.5" />
    <line :x1="x(v1)" :x2="x(v1)" :y1="m.t - 14" :y2="m.t + 7 * fila" stroke="#1A2133" stroke-width="1.8" />
    <text :x="x(v0) + 6" :y="m.t - 14" style="font-size: 16px">v(∅) = {{ v0 }}</text>
    <text :x="x(v1) - 6" :y="m.t - 14" text-anchor="end" style="font-size: 16px">v(todos) = f(x) = {{ v1 }}</text>
    <g v-for="(f, i) in filas" :key="i" :transform="`translate(0, ${m.t + i * fila + (f.shapley ? 8 : 0)})`">
      <text :x="m.l - 14" y="31" text-anchor="end" :style="{ fill: '#1A2133', fontWeight: f.shapley ? 700 : 400 }">{{ f.t }}</text>
      <g v-for="s in tramos[i]" :key="s.k">
        <rect :x="s.x0" y="8" :width="s.x1 - s.x0 - 1.5" height="32" :fill="COL[s.k]" :opacity="f.shapley ? 1 : 0.88" rx="2" />
        <text :x="(s.x0 + s.x1) / 2" y="30" text-anchor="middle" style="fill: #fff; font-weight: 650; font-size: 17px">{{ s.c.toFixed(1) }}</text>
      </g>
    </g>
    <g class="eje">
      <text v-for="t in [180, 200, 220, 240, 260, 280, 300, 320]" :key="t" :x="x(t)" :y="H - 30" text-anchor="middle" style="font-size: 16px">{{ t }}</text>
      <text class="titulo-eje" :x="(m.l + W - m.r) / 2" :y="H - 4" text-anchor="middle" style="font-size: 17px">predicción (miles de dólares)</text>
    </g>
    <g :transform="`translate(${m.l - 236}, ${H - 40})`">
      <template v-for="(k, i) in ['area', 'calidad', 'anio']" :key="k">
        <rect :x="i * 75" y="-13" width="14" height="14" :fill="COL[k]" rx="2" />
        <text :x="i * 75 + 19" y="0" style="font-size: 16px; fill: #1A2133">{{ NOM[k] }}</text>
      </template>
    </g>
  </svg>
</template>
