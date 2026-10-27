<script setup lang="ts">
// PSI de la temperatura por mes: contra todo 2011 (la trampa) y contra el mismo mes de 2011.
import { computed } from 'vue'
import { scaleLinear, scaleBand } from 'd3-scale'
import d from '../datos/u08-02.json'
const P = (d as any).psi_temp
const props = withDefaults(defineProps<{ modo?: 'ingenuo' | 'ambos' }>(), { modo: 'ambos' })
const W = 1100, H = 400, m = { l: 70, r: 60, t: 76, b: 50 }
const meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const x = scaleBand().domain(meses).range([m.l, W - m.r]).paddingInner(0.25).paddingOuter(0.1)
const y = scaleLinear().domain([0, 5.4]).range([H - m.b, m.t])
const ancho = computed(() => props.modo === 'ambos' ? x.bandwidth() / 2 - 2 : x.bandwidth())
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <g class="rejilla"><line v-for="t in [1, 2, 3, 4, 5]" :key="t" :x1="m.l" :x2="W - m.r" :y1="y(t)" :y2="y(t)" /></g>
    <g class="eje">
      <line :x1="m.l" :x2="W - m.r" :y1="y(0)" :y2="y(0)" />
      <text v-for="t in [0, 1, 2, 3, 4, 5]" :key="`y${t}`" :x="m.l - 12" :y="y(t) + 6" text-anchor="end">{{ t }}</text>
      <text v-for="mm in meses" :key="mm" :x="x(mm)! + x.bandwidth() / 2" :y="H - m.b + 28" text-anchor="middle">{{ mm }}</text>
      <text class="titulo-eje" :x="m.l - 50" :y="m.t - 22">PSI de la temperatura</text>
    </g>
    <g v-for="(mm, i) in meses" :key="`b${mm}`">
      <rect :x="x(mm)" :y="y(P.ingenuo2011[i])" :width="ancho" :height="y(0) - y(P.ingenuo2011[i])" rx="2" fill="#8A91A0" />
      <template v-if="modo === 'ambos'">
        <rect :x="x(mm)! + x.bandwidth() / 2 + 2" :y="y(P.mismo_mes[i])" :width="ancho" :height="y(0) - y(P.mismo_mes[i])" rx="2" fill="#24398A" />
        <text :x="x(mm)! + 3 * x.bandwidth() / 4 + 1" :y="y(P.mismo_mes[i]) - 8" text-anchor="middle" style="font-size: 15px; fill: #1A2133">{{ P.mismo_mes[i].toFixed(2) }}</text>
      </template>
    </g>
    <line :x1="m.l" :x2="W - m.r" :y1="y(0.25)" :y2="y(0.25)" stroke="#D5433C" stroke-width="1.6" stroke-dasharray="7 5" />
    <text :x="W - m.r + 6" :y="y(0.25) + 6" style="fill: #D5433C; font-size: 16px; font-weight: 600">0.25</text>
    <g :transform="`translate(${m.l + 200}, 16)`">
      <rect x="0" y="-14" width="18" height="18" rx="2" fill="#8A91A0" />
      <text x="28" y="2" style="fill:#1A2133">cada mes de 2011 contra todo 2011</text>
      <template v-if="modo === 'ambos'">
        <rect x="380" y="-14" width="18" height="18" rx="2" fill="#24398A" />
        <text x="408" y="2" style="fill:#1A2133">cada mes de 2012 contra el mismo mes de 2011</text>
      </template>
    </g>
  </svg>
</template>
