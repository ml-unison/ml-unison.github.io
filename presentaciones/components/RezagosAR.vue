<script setup lang="ts">
// Un AR(2) simulado: para cada tiempo t, los dos valores anteriores son los atributos y y_t es la respuesta. Deslizador para mover t.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-02.json'
const A = (datos as any).ar2
const N = 48
const s: number[] = A.serie.slice(0, N)
const t = ref(20)
const W = 700, H = 300, m = { l: 50, r: 14, t: 20, b: 44 }
const x = scaleLinear().domain([0.5, N + 0.5]).range([m.l, W - m.r])
const y = scaleLinear().domain([-1.5, 6]).range([H - m.b, m.t])
const camino = line<number>().x((_, i) => x(i + 1)).y(v => y(v))(s)!
const f = (v: number) => v.toFixed(2).replace('-', '−')
const pred = computed(() => A.mc[0] + A.mc[1] * s[t.value - 2] + A.mc[2] * s[t.value - 3])
</script>

<template>
  <div>
    <div class="deslizador"><span>t =</span><input type="range" min="3" :max="N" v-model.number="t" /><output>{{ t }}</output></div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <g class="rejilla"><line v-for="v in [0, 2, 4, 6]" :key="v" :x1="m.l" :x2="W - m.r" :y1="y(v)" :y2="y(v)" /></g>
      <g class="eje">
        <text v-for="v in [0, 2, 4, 6]" :key="v" :x="m.l - 10" :y="y(v) + 6" text-anchor="end">{{ v }}</text>
        <text v-for="v in [1, 10, 20, 30, 40]" :key="`x${v}`" :x="x(v)" :y="H - m.b + 26" text-anchor="middle">{{ v }}</text>
      </g>
      <rect :x="x(t - 2.5)" :y="m.t" :width="x(t + 0.5) - x(t - 2.5)" :height="H - m.b - m.t" fill="#EFF2F9" rx="6" />
      <path :d="camino" fill="none" stroke="#8A91A0" stroke-width="2" />
      <circle v-for="(v, i) in s" :key="i" :cx="x(i + 1)" :cy="y(v)" r="4" fill="#8A91A0" />
      <circle :cx="x(t - 1)" :cy="y(s[t - 2])" r="8" fill="#24398A" />
      <circle :cx="x(t - 2)" :cy="y(s[t - 3])" r="8" fill="#24398A" />
      <circle :cx="x(t)" :cy="y(s[t - 1])" r="8" fill="#D5433C" />
      <line :x1="x(t) - 14" :x2="x(t) + 14" :y1="y(pred)" :y2="y(pred)" stroke="#C9971C" stroke-width="4" />
    </svg>
    <table class="ra">
      <tr class="enc"><td>y<sub>t−1</sub></td><td>y<sub>t−2</sub></td><td></td><td>y<sub>t</sub></td><td>ŷ<sub>t</sub></td></tr>
      <tr><td class="az">{{ f(s[t - 2]) }}</td><td class="az">{{ f(s[t - 3]) }}</td><td>→</td><td class="ro">{{ f(s[t - 1]) }}</td><td class="do">{{ f(pred) }}</td></tr>
    </table>
  </div>
</template>

<style scoped>
.ra { width: auto !important; font-size: 21px !important; margin-top: 4px; }
.ra .enc td { color: var(--tenue) !important; font-style: italic; border-bottom: 2px solid var(--marino); }
.ra td { padding: 4px 22px 4px 0 !important; text-align: right; font-variant-numeric: tabular-nums; }
.ra .az { color: #24398A; font-weight: 650; }
.ra .ro { color: #D5433C; font-weight: 650; }
.ra .do { color: #A87C12; font-weight: 650; }
</style>
