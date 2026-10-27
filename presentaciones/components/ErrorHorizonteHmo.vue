<script setup lang="ts">
// MAE en 2021–2024 contra el horizonte para cuatro pronósticos de referencia, con un deslizador para leer un horizonte.
import { ref } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-01.json'
const R = (datos as any).referencias
const W = 760, H = 420, m = { l: 62, r: 200, t: 30, b: 64 }
const x = scaleLinear().domain([1, 30]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 4]).range([H - m.b, m.t])
const metodos = [
  { n: 'ingenuo', k: 'ingenuo', c: '#D5433C' },
  { n: 'estacional ingenuo', k: 'estacional ingenuo', c: '#6B7385' },
  { n: 'climatología', k: 'climatología', c: '#24398A' },
  { n: 'AR(1)', k: 'AR(1) de anomalías', c: '#C9971C' },
].map(o => ({ ...o, d: R.mae_h[o.k] as number[], p: line<number>().x((_, i) => x(i + 1)).y(v => y(v))(R.mae_h[o.k])! }))
const h = ref(1)
// etiquetas a la derecha, separadas al menos 26 px
const etiquetas = (() => {
  const u = metodos.map(o => ({ n: o.n, c: o.c, y: y(o.d[29]) })).sort((a, b) => a.y - b.y)
  for (let i = 1; i < u.length; i++) u[i].y = Math.max(u[i].y, u[i - 1].y + 26)
  return u
})()
</script>

<template>
  <div>
    <div class="deslizador">
      <span>horizonte h =</span>
      <input type="range" min="1" max="30" step="1" v-model.number="h" />
      <output>{{ h }}</output><span class="tenue pequeno">{{ h === 1 ? 'día' : 'días' }}</span>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
      <Ejes :x="x" :y="y" :x-ticks="[1, 5, 10, 15, 20, 25, 30]" :y-ticks="[0, 1, 2, 3, 4]" x-titulo="horizonte h (días)" y-titulo="MAE en 2021–2024 (°C)" />
      <line :x1="x(h)" :x2="x(h)" :y1="m.t" :y2="H - m.b" stroke="#8A91A0" stroke-dasharray="5 5" />
      <path v-for="o in metodos" :key="o.n" :d="o.p" fill="none" :stroke="o.c" stroke-width="3.4" />
      <g v-for="o in metodos" :key="`c${o.n}`">
        <circle :cx="x(h)" :cy="y(o.d[h - 1])" r="6" :fill="o.c" stroke="#fff" stroke-width="1.5" />
      </g>
      <g v-for="e in etiquetas" :key="`e${e.n}`">
        <line :x1="W - m.r + 10" :x2="W - m.r + 28" :y1="e.y" :y2="e.y" :stroke="e.c" stroke-width="4" />
        <text :x="W - m.r + 34" :y="e.y + 6" style="fill: #1A2133; font-weight: 600">{{ e.n }} {{ metodos.find(o => o.n === e.n)!.d[h - 1].toFixed(2) }}</text>
      </g>
    </svg>
  </div>
</template>
