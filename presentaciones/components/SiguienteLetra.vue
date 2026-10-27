<script setup lang="ts">
// Distribución del bigrama de Laplace para la letra que sigue a «q» y a un espacio.
import { scaleBand, scaleLinear } from 'd3-scale'
import datos from '../datos/u06-04.json'
const S = (datos as any).siguiente
const W = 560, H = 190, m = { l: 46, r: 8, t: 14, b: 40 }
const vis = (c: string) => c === ' ' ? '·' : c === '\n' ? '¶' : c
const paneles = [
  { t: 'después de «q»', d: S.q },
  { t: 'después de un espacio «·»', d: S.espacio },
].map(p => {
  const n = 10
  const x = scaleBand<number>().domain([...Array(n).keys()]).range([m.l, W - m.r]).padding(0.18)
  const y = scaleLinear().domain([0, Math.max(...p.d.p.slice(0, n))]).nice(4).range([H - m.b, m.t])
  return { ...p, n, x, y }
})
</script>

<template>
  <div class="sl">
    <div v-for="p in paneles" :key="p.t">
      <div class="sl__t">{{ p.t }} · <span>perplejidad local {{ p.d.ppl.toFixed(1) }}</span></div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <g class="rejilla"><line v-for="t in p.y.ticks(3)" :key="t" :x1="m.l" :x2="W - m.r" :y1="p.y(t)" :y2="p.y(t)" /></g>
        <text v-for="t in p.y.ticks(3)" :key="`t${t}`" :x="m.l - 8" :y="p.y(t) + 6" text-anchor="end">{{ t }}</text>
        <g v-for="i in p.n" :key="i">
          <rect :x="p.x(i - 1)" :y="p.y(p.d.p[i - 1])" :width="p.x.bandwidth()" :height="p.y(0) - p.y(p.d.p[i - 1])" fill="#24398A" />
          <text :x="p.x(i - 1)! + p.x.bandwidth() / 2" :y="H - m.b + 28" text-anchor="middle" style="fill: #1A2133; font-size: 22px; font-weight: 600">{{ vis(p.d.letras[i - 1]) }}</text>
        </g>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.sl { display: flex; flex-direction: column; gap: 6px; }
.sl__t { font-weight: 600; font-size: 20px; color: var(--marino); }
.sl__t span { font-weight: 400; color: var(--tenue); }
</style>
