<script setup lang="ts">
// Error relativo (escala log) en el valle con κ = 50: misma η con varios β, y la pareja (η, β) óptima.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-02.json'
const V = (datos as any).valle
const W = 560, H = 400, m = { l: 70, r: 20, t: 36, b: 64 }
const x = scaleLinear().domain([0, 400]).range([m.l, W - m.r])
const y = scaleLinear().domain([-9, 0.3]).range([H - m.b, m.t])
const colores: Record<string, string> = { '0': '#D5433C', '0.5': '#6B7385', '0.64': '#24398A', '0.9': '#C9971C' }
const camino = (e: number[]) => line<number>().x((_, i) => x(i)).y(v => y(v))(e.slice(0, 401))!
const curvas = V.caminos.map((c: any) => ({ beta: c.beta, iter: c.iter, d: camino(c.error), color: colores[String(c.beta)] }))
const etiq: Record<string, [number, number]> = { '0': [300, -4.0], '0.5': [74, -5.1], '0.64': [124, -8.4], '0.9': [240, -2.1] }
const potencia = (v: number) => v === 0 ? '1' : `10${String(v).replace('-', '⁻').replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d])}`
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 100, 200, 300, 400]" :y-ticks="[0, -3, -6, -9]" :y-formato="potencia" x-titulo="iteración" y-titulo="error relativo ‖w_t − w*‖ / ‖w_0 − w*‖" />
    <line :x1="m.l" :x2="W - m.r" :y1="y(-6)" :y2="y(-6)" stroke="#5E6678" stroke-dasharray="4 4" />
    <clipPath id="em-clip"><rect :x="m.l" :y="m.t - 10" :width="W - m.l - m.r" :height="H - m.t - m.b + 10" /></clipPath>
    <g clip-path="url(#em-clip)">
    <path :d="camino(V.optimo.error)" fill="none" stroke="#24398A" stroke-width="2.4" stroke-dasharray="3 5" />
    <path v-for="c in curvas" :key="c.beta" :d="c.d" fill="none" :stroke="c.color" stroke-width="2.8" />
    </g>
    <text v-for="c in curvas" :key="`t${c.beta}`" :x="x(etiq[String(c.beta)][0])" :y="y(etiq[String(c.beta)][1])" :style="{ fill: c.color, fontWeight: 600, fontSize: '18px' }">
      {{ c.beta === 0 ? 'sin momento' : `β = ${c.beta}` }}
    </text>
  </svg>
</template>
