<script setup lang="ts">
// ACF muestral de la temperatura cruda (0–730 días) y de las anomalías (0–30 días), con la banda ±1.96/√T.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-01.json'
const props = withDefaults(defineProps<{ anomalias?: boolean }>(), { anomalias: true })
const A = (datos as any).acf
const W = 540, H = 380, m = { l: 62, r: 16, t: 40, b: 64 }
const y = scaleLinear().domain([-0.85, 1]).range([H - m.b, m.t])
const xc = scaleLinear().domain([0, 730]).range([m.l, W - m.r])
const xa = scaleLinear().domain([-0.5, 30.5]).range([m.l, W - m.r])
const cruda = line<number>().x((_, i) => xc(i)).y(v => y(v))(A.cruda)!
</script>

<template>
  <div class="acfh">
    <div>
      <div class="acfh__t">Temperatura</div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
        <Ejes :x="xc" :y="y" :x-ticks="[0, 182, 365, 547, 730]" :y-ticks="[-0.5, 0, 0.5, 1]" x-titulo="rezago k (días)" y-titulo="ρ̂(k)" />
        <rect :x="m.l" :width="W - m.l - m.r" :y="y(A.banda)" :height="y(-A.banda) - y(A.banda)" fill="#C0CEE3" />
        <line :x1="m.l" :x2="W - m.r" :y1="y(0)" :y2="y(0)" stroke="#8A91A0" />
        <path :d="cruda" fill="none" stroke="#24398A" stroke-width="3.5" />
        <text :x="xc(234)" :y="y(-0.7)" text-anchor="start" style="fill: #1A2133">enero contra julio</text>
        <text :x="xc(365)" :y="y(0.79) - 12" text-anchor="middle" style="fill: #1A2133">un año</text>
      </svg>
    </div>
    <div v-if="anomalias">
      <div class="acfh__t">Anomalías</div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
        <Ejes :x="xa" :y="y" :x-ticks="[0, 7, 14, 21, 28]" :y-ticks="[-0.5, 0, 0.5, 1]" x-titulo="rezago k (días)" y-titulo="ρ̂(k)" />
        <rect :x="m.l" :width="W - m.l - m.r" :y="y(A.banda)" :height="y(-A.banda) - y(A.banda)" fill="#C0CEE3" />
        <line :x1="m.l" :x2="W - m.r" :y1="y(0)" :y2="y(0)" stroke="#8A91A0" />
        <g v-for="(v, k) in A.anom" :key="k">
          <line :x1="xa(k)" :x2="xa(k)" :y1="y(0)" :y2="y(v)" stroke="#D5433C" stroke-width="3.5" />
          <circle :cx="xa(k)" :cy="y(v)" r="4.5" fill="#D5433C" />
        </g>
        <text :x="xa(1) + 12" :y="y(A.anom[1]) + 4" style="fill: #1A2133; font-weight: 600">{{ A.anom[1].toFixed(2) }}</text>
        <text :x="xa(7)" :y="y(A.anom[7]) - 14" text-anchor="middle" style="fill: #1A2133; font-weight: 600">{{ A.anom[7].toFixed(2) }}</text>
        <text :x="xa(30)" :y="y(A.anom[30]) - 14" text-anchor="end" style="fill: #1A2133; font-weight: 600">{{ A.anom[30].toFixed(2) }}</text>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.acfh { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
.acfh__t { font-weight: 650; font-size: 21px; color: var(--marino); margin-bottom: 2px; }
</style>
