<script setup lang="ts">
// Matriz de confusión de la red convolucional en las 10,000 imágenes de prueba de Fashion-MNIST.
import { scaleBand } from 'd3-scale'
import datos from '../datos/u06-03.json'
const D = datos as any
const M: number[][] = D.confusion
const W = 600, H = 560, m = { l: 140, r: 8, t: 8, b: 130 }
const x = scaleBand<number>().domain([...Array(10).keys()]).range([m.l, W - m.r])
const y = scaleBand<number>().domain([...Array(10).keys()]).range([m.t, H - m.b])
const fondo = (i: number, j: number) => {
  const v = M[i][j]
  if (i === j) return `rgba(36,57,138,${0.15 + 0.6 * v / 1000})`
  return v === 0 ? '#fff' : `rgba(213,67,60,${Math.min(1, 0.08 + v / 160)})`
}
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" style="max-height: 530px">
    <g v-for="i in 10" :key="i">
      <g v-for="j in 10" :key="j">
        <rect :x="x(j - 1)" :y="y(i - 1)" :width="x.bandwidth() - 2" :height="y.bandwidth() - 2" :fill="fondo(i - 1, j - 1)" rx="2" />
        <text v-if="M[i - 1][j - 1] >= 5" :x="x(j - 1)! + x.bandwidth() / 2 - 1" :y="y(i - 1)! + y.bandwidth() / 2 + 6" text-anchor="middle"
              :style="{ fontSize: '15px', fill: (i === j || M[i - 1][j - 1] > 100) ? '#fff' : '#1A2133', fontWeight: M[i - 1][j - 1] > 100 && i !== j ? 700 : 400 }">{{ M[i - 1][j - 1] }}</text>
      </g>
      <text :x="m.l - 10" :y="y(i - 1)! + y.bandwidth() / 2 + 6" text-anchor="end" style="font-size: 17px; fill: #1A2133">{{ D.clases[i - 1] }}</text>
      <text :transform="`translate(${x(i - 1)! + x.bandwidth() / 2 + 5}, ${H - m.b + 10}) rotate(-50)`" text-anchor="end" style="font-size: 17px; fill: #1A2133">{{ D.clases[i - 1] }}</text>
    </g>
  </svg>
</template>
