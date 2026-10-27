<script setup lang="ts">
// Error relativo de la derivada numérica de un peso contra el paso h (diferencia hacia adelante y centrada):
// la «V» entre el error de truncamiento y el de redondeo. Datos: herramientas/laminas/u06_01.py.
import { scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-01.json'
const D = (datos as any).difs
const W = 620, H = 420, m = { l: 70, r: 16, t: 40, b: 66 }
const x = scaleLog().domain([0.15, 7e-14]).range([m.l, W - m.r]), y = scaleLog().domain([2e-11, 1]).range([H - m.b, m.t])
const pot = (e: number) => '10' + String(e).replace('-', '⁻').replace(/\d/g, c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c])
const camino = (v: number[]) => line<number>().x((_, i) => x(D.h[i])).y(t => y(Math.max(t, 2e-11)))(v)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[1e-1, 1e-4, 1e-7, 1e-10, 1e-13]" :y-ticks="[1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]"
          :x-formato="v => pot(Math.round(Math.log10(v)))" :y-formato="v => v === 1 ? '1' : pot(Math.round(Math.log10(v)))"
          x-titulo="paso h (cada vez más chico →)" y-titulo="error relativo" />
    <path :d="camino(D.adelante)" fill="none" stroke="#D5433C" stroke-width="3" />
    <path :d="camino(D.centrada)" fill="none" stroke="#24398A" stroke-width="3" />
    <text :x="x(0.08)" :y="y(0.25)" style="fill: #1A2133; font-weight: 600">hacia adelante: O(h)</text>
    <text :x="x(0.1)" :y="y(4e-10)" style="fill: #1A2133; font-weight: 600">centrada: O(h²)</text>
    <text :x="x(3e-11)" :y="y(1e-8)" text-anchor="middle" style="font-size: 17px">aquí domina el</text>
    <text :x="x(3e-11)" :y="y(1e-8) + 22" text-anchor="middle" style="font-size: 17px">redondeo: ε / h</text>
  </svg>
</template>
