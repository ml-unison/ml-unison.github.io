<script setup lang="ts">
// Cascada SHAP de una casa de prueba: factores que multiplican el precio de referencia (eje log en dólares).
import { scaleLog } from 'd3-scale'
import datos from '../datos/u05-04.json'
import { NOMBRES, UNIDADES } from './casasAmes'
const C = (datos as any).cascada
const k = 7
const orden = Object.keys(C.phi).sort((a, b) => Math.abs(C.phi[b]) - Math.abs(C.phi[a]))
const resto = orden.slice(k)
const fmt = (c: string) => { const v = C.x[c]; return `${NOMBRES[c]} = ${c === 'area' || c === 'sotano' ? Math.round(v) : v}${UNIDADES[c] ?? ''}` }
const pasos = [...orden.slice(0, k).map(c => ({ t: fmt(c), p: C.phi[c] })), { t: `otros ${resto.length} atributos`, p: resto.reduce((s, c) => s + C.phi[c], 0) }]
const W = 1060, fila = 40, m = { l: 330, r: 40, t: 40, b: 62 }
const x = scaleLog().domain([150, 290]).range([m.l, W - m.r])
const P = (z: number) => x(Math.exp(z) / 1000)
let nivel = C.phi0
const barras = pasos.map(s => { const b = { ...s, a: nivel, b: nivel + s.p }; nivel += s.p; return b })
const final = nivel
const H = m.t + pasos.length * fila + m.b
const ticks = [150, 175, 200, 225, 250, 275]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <g class="rejilla"><line v-for="t in ticks" :key="t" :x1="x(t)" :x2="x(t)" :y1="m.t - 6" :y2="m.t + pasos.length * fila" /></g>
    <line :x1="P(C.phi0)" :x2="P(C.phi0)" :y1="m.t - 18" :y2="m.t + pasos.length * fila" stroke="#5E6678" stroke-dasharray="5 4" stroke-width="1.6" />
    <line :x1="P(final)" :x2="P(final)" :y1="m.t - 18" :y2="m.t + pasos.length * fila" stroke="#1A2133" stroke-width="2" />
    <text :x="P(C.phi0) - 6" :y="m.t - 20" text-anchor="end" style="font-size: 17px">referencia: {{ (Math.exp(C.phi0) / 1000).toFixed(0) }} mil</text>
    <text :x="P(final) - 8" :y="m.t - 20" text-anchor="end" style="font-size: 17px; fill: #1A2133; font-weight: 650">predicción: {{ C.pred.toFixed(0) }} mil (real: {{ C.real.toFixed(0) }} mil)</text>
    <g v-for="(b, i) in barras" :key="i" :transform="`translate(0, ${m.t + i * fila})`">
      <text :x="m.l - 14" y="26" text-anchor="end" style="fill: #1A2133">{{ b.t }}</text>
      <rect :x="Math.min(P(b.a), P(b.b))" y="6" :width="Math.abs(P(b.b) - P(b.a))" height="28" :fill="b.p > 0 ? '#D5433C' : '#24398A'" rx="2" />
      <text :x="b.p > 0 ? P(b.b) + 8 : P(b.b) - 8" y="26" :text-anchor="b.p > 0 ? 'start' : 'end'" style="fill: #1A2133; font-weight: 650; font-size: 18px">×{{ Math.exp(b.p).toFixed(2) }}</text>
    </g>
    <g class="eje">
      <line :x1="m.l" :x2="W - m.r" :y1="m.t + pasos.length * fila" :y2="m.t + pasos.length * fila" />
      <text v-for="t in ticks" :key="t" :x="x(t)" :y="H - 34" text-anchor="middle">{{ t }}</text>
      <text class="titulo-eje" :x="(m.l + W - m.r) / 2" :y="H - 6" text-anchor="middle">precio (miles de dólares, escala logarítmica)</text>
    </g>
  </svg>
</template>
