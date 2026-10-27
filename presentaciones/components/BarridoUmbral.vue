<script setup lang="ts">
// Barrido de todos los umbrales del largo del pico en la raíz: conteos a cada lado y ganancia de Gini.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import { SERIE } from './colores'
import datos from '../datos/u05-01.json'
const B = (datos as any).barrido, P = (datos as any).pinguinos
const U = B.umbral as number[]
const iMax = B.gan.indexOf(Math.max(...B.gan))
const sel = ref(U.findIndex((u: number) => u >= 38))
const W = 660, H = 430, m = { l: 64, r: 16, t: 24, b: 60 }, base = 300
const x = scaleLinear().domain([31, 61]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 0.32]).range([base - 20, m.t])
const curva = line<number>().x((_, i) => x(U[i])).y(g => y(g))(B.gan)!
const ESP = ['Adelie', 'Chinstrap', 'Gentoo']
const gini = (c: number[]) => { const n = c.reduce((a, b) => a + b, 0); return n ? 1 - c.reduce((a, b) => a + (b / n) ** 2, 0) : 0 }
const izq = computed(() => B.izq[sel.value] as number[])
const der = computed(() => B.total.map((t: number, k: number) => t - izq.value[k]))
const u = computed(() => U[sel.value])
// jitter determinista para la tira de puntos
const sac = (P.x as number[]).map((_, i) => ((i * 7919) % 97) / 97 - 0.5)
</script>

<template>
  <div class="bu">
    <div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <rect :x="m.l" :y="m.t" :width="x(u) - m.l" :height="H - m.b - m.t" fill="#EFF2F9" />
        <g class="rejilla"><line v-for="t in [0, 0.1, 0.2, 0.3]" :key="t" :x1="m.l" :x2="W - m.r" :y1="y(t)" :y2="y(t)" /></g>
        <g class="eje">
          <text v-for="t in [0, 0.1, 0.2, 0.3]" :key="`y${t}`" :x="m.l - 10" :y="y(t) + 6" text-anchor="end">{{ t }}</text>
          <line :x1="m.l" :x2="W - m.r" :y1="H - m.b" :y2="H - m.b" />
          <text v-for="t in [35, 40, 45, 50, 55, 60]" :key="`x${t}`" :x="x(t)" :y="H - m.b + 26" text-anchor="middle">{{ t }}</text>
          <text class="titulo-eje" :x="(m.l + W - m.r) / 2" :y="H - 6" text-anchor="middle">umbral t sobre el largo del pico (mm)</text>
          <text class="titulo-eje" :x="m.l" :y="m.t - 6">ganancia de Gini</text>
        </g>
        <path :d="curva" fill="none" stroke="#24398A" stroke-width="3" />
        <g v-for="k in [0, 1, 2]" :key="k">
          <circle v-for="(v, i) in P.x" v-show="P.g[i] === k" :key="i" :cx="x(v)" :cy="base + 22 + 26 * k + 14 * sac[i]" r="3.4" :fill="SERIE[k]" opacity="0.8" />
        </g>
        <line :x1="x(u)" :x2="x(u)" :y1="m.t" :y2="H - m.b" stroke="#1A2133" stroke-width="2" />
        <circle :cx="x(u)" :cy="y(B.gan[sel])" r="7" fill="#fff" stroke="#24398A" stroke-width="3" />
        <text :x="x(U[iMax]) + 8" :y="y(B.gan[iMax]) - 4" style="font-size: 16px">máximo en {{ U[iMax].toFixed(2) }}</text>
      </svg>
      <div class="deslizador">
        <span>t</span>
        <input v-model.number="sel" type="range" min="0" :max="U.length - 1" step="1">
        <output>{{ u.toFixed(2) }} mm</output>
      </div>
    </div>
    <div class="bu__lado">
      <table>
        <tr><th></th><th>x ≤ t</th><th>x &gt; t</th></tr>
        <tr v-for="(e, k) in ESP" :key="e"><td :style="{ color: SERIE[k] }">{{ e }}</td><td>{{ izq[k] }}</td><td>{{ der[k] }}</td></tr>
        <tr><td>Gini</td><td>{{ gini(izq).toFixed(3) }}</td><td>{{ gini(der).toFixed(3) }}</td></tr>
      </table>
      <div class="bu__gan">ganancia = {{ B.padre.toFixed(3) }} − promedio ponderado = <b>{{ B.gan[sel].toFixed(3) }}</b></div>
      <slot />
    </div>
  </div>
</template>

<style scoped>
.bu { display: grid; grid-template-columns: 660px 1fr; gap: 36px; align-items: start; }
.bu__lado { font-size: 20px; display: flex; flex-direction: column; gap: 16px; }
.bu table { font-size: 20px; }
.bu table th { font-size: 15px; padding: 0 12px 8px 0; }
.bu table td { padding: 7px 12px 7px 0; font-variant-numeric: tabular-nums; }
.bu__gan { font-size: 19px; color: var(--tenue); }
.bu__gan b { color: var(--azul); }
</style>
