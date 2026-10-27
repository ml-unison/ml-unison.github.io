<script setup lang="ts">
// Histograma a partir de cuentas y bordes precalculados (Python).
import { scaleLinear } from 'd3-scale'
const props = withDefaults(defineProps<{ cuentas: number[], bordes: number[], titulo?: string, xTitulo?: string, ancho?: number, alto?: number, xTicks?: number[] }>(),
  { ancho: 540, alto: 320 })
const m = { l: 44, r: 14, t: 34, b: 58 }
const x = scaleLinear().domain([props.bordes[0], props.bordes[props.bordes.length - 1]]).range([m.l, props.ancho - m.r])
const y = scaleLinear().domain([0, Math.max(...props.cuentas) * 1.08]).range([props.alto - m.b, m.t])
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${alto}`" role="img">
    <text v-if="titulo" :x="m.l" y="20" style="fill:#20305A; font-weight: 650; font-size: 20px">{{ titulo }}</text>
    <rect v-for="(c, i) in cuentas" :key="i" :x="x(bordes[i]) + 1" :y="y(c)" :width="Math.max(0, x(bordes[i + 1]) - x(bordes[i]) - 2)" :height="y(0) - y(c)" rx="1.5" fill="#24398A" />
    <g class="eje">
      <line :x1="m.l" :x2="ancho - m.r" :y1="y(0)" :y2="y(0)" />
      <text v-for="t in (xTicks ?? x.ticks(5))" :key="t" :x="x(t)" :y="y(0) + 24" text-anchor="middle">{{ t }}</text>
      <text v-if="xTitulo" class="titulo-eje" :x="(m.l + ancho - m.r) / 2" :y="alto - 6" text-anchor="middle">{{ xTitulo }}</text>
    </g>
  </svg>
</template>
