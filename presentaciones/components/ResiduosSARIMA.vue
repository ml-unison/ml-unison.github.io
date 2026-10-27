<script setup lang="ts">
// Residuos del SARIMA(1,1,1)(0,1,1)₁₂ en entrenamiento: la serie y su ACF.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-02.json'
const R = (datos as any).residuos
const W = 560, H = 230, m = { l: 58, r: 10, t: 34, b: 40 }
const x = scaleLinear().domain([1959, 2015]).range([m.l, W - m.r])
const y = scaleLinear().domain([-1.2, 1.2]).range([H - m.b, m.t])
const camino = line<number>().x((_, i) => x(R.t[i])).y(v => y(v))(R.y)!
</script>

<template>
  <div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <text :x="m.l" y="18" style="fill: #1A2133; font-weight: 600; font-size: 19px">residuos (ppm)</text>
      <g class="rejilla"><line v-for="t in [-1, 0, 1]" :key="t" :x1="m.l" :x2="W - m.r" :y1="y(t)" :y2="y(t)" /></g>
      <g class="eje">
        <text v-for="t in [1960, 1980, 2000]" :key="t" :x="x(t)" :y="H - m.b + 24" text-anchor="middle">{{ t }}</text>
        <text v-for="t in [-1, 0, 1]" :key="`y${t}`" :x="m.l - 10" :y="y(t) + 6" text-anchor="end">{{ String(t).replace('-', '−') }}</text>
      </g>
      <path :d="camino" fill="none" stroke="#24398A" stroke-width="0.9" />
    </svg>
    <CorrelogramaARIMA :valores="R.acf" :banda="R.banda" :periodo="12" titulo="ACF de los residuos" :ancho="560" :alto="230" :y-dominio="[-0.5, 0.5]" :y-ticks="[-0.5, 0, 0.5]" />
  </div>
</template>
