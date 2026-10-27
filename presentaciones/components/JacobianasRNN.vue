<script setup lang="ts">
// ‖∂h_T/∂h_t‖ en una RNN tanh de 256 unidades con W = ρQ (Q ortogonal), en escala logarítmica.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-04.json'
const R = (datos as any).rnn
const W = 620, H = 420, m = { l: 70, r: 90, t: 34, b: 70 }
const x = scaleLinear().domain([0, 100]).range([m.l, W - m.r])
const y = scaleLinear().domain([-21, 14]).range([H - m.b, m.t])
const colores: Record<string, string> = { '0.8': '#D5433C', '1.0': '#6B7385', '1.5': '#C9971C', '3.0': '#24398A' }
const series = Object.keys(R).sort((a, b) => Number(a) - Number(b)).map(k => ({
  k, c: colores[k], v: R[k] as number[],
  d: line<number>().x((_, i) => x(i)).y(v => y(v))(R[k])!,
}))
const pot = (v: number) => v === 0 ? '1' : `10${String(v).replace('-', '⁻').replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d])}`
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 20, 40, 60, 80, 100]" :y-ticks="[-20, -15, -10, -5, 0, 5, 10]" :y-formato="pot" x-titulo="distancia T − t (letras hacia atrás)" y-titulo="‖∂hT / ∂ht‖" />
    <path v-for="s in series" :key="s.k" :d="s.d" fill="none" :stroke="s.c" stroke-width="3" />
    <text v-for="s in series" :key="`t${s.k}`" :x="x(s.v.length - 1) + 10" :y="y(s.v[s.v.length - 1]) + 6" class="etiqueta" style="fill: #1A2133">ρ = {{ s.k }}</text>
  </svg>
</template>
