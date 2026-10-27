<script setup lang="ts">
// Anomalía media anual de la temperatura de Hermosillo (respecto a 1992–2020) y su recta con banda de 95 %.
import { scaleLinear } from 'd3-scale'
import { area, line } from 'd3-shape'
import datos from '../datos/u07-01.json'
const T = (datos as any).tendencia
const W = 760, H = 400, m = { l: 62, r: 20, t: 36, b: 50 }
const x = scaleLinear().domain([1991.3, 2024.7]).range([m.l, W - m.r]), y = scaleLinear().domain([-1.5, 2.3]).range([H - m.b, m.t])
const bw = (x(1) - x(0)) * 0.72
const banda = area<number>().x((_, i) => x(T.malla[i])).y0((_, i) => y(T.lo[i])).y1((_, i) => y(T.hi[i]))(T.malla)!
const recta = line<number>().x((_, i) => x(T.malla[i])).y(v => y(v))(T.recta)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <Ejes :x="x" :y="y" :x-ticks="[1995, 2000, 2005, 2010, 2015, 2020]" :x-formato="v => String(v)" :y-ticks="[-1, 0, 1, 2]" y-titulo="°C respecto a la climatología 1992–2020" />
    <rect v-for="(a, i) in T.anio" :key="a" :x="x(a) - bw / 2" :width="bw" :y="Math.min(y(0), y(T.anom[i]))" :height="Math.abs(y(T.anom[i]) - y(0))"
          :fill="T.anom[i] > 0 ? '#D5433C' : '#24398A'" opacity="0.9" />
    <line :x1="m.l" :x2="W - m.r" :y1="y(0)" :y2="y(0)" stroke="#8A91A0" />
    <path :d="banda" fill="#6B7385" opacity="0.22" />
    <path :d="recta" fill="none" stroke="#1A2133" stroke-width="3" />
    <text :x="x(2014)" :y="y(T.anom[22]) - 10" text-anchor="middle" style="fill: #1A2133; font-weight: 600">2014</text>
    <text :x="x(2006)" :y="y(T.anom[14]) + 26" text-anchor="middle" style="fill: #1A2133; font-weight: 600">2006</text>
  </svg>
</template>
