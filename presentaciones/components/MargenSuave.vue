<script setup lang="ts">
// Adelie contra Chinstrap (largo y profundidad del pico, estandarizados): margen suave para varios C.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { recta } from './recta'
import datos from '../datos/u04.json'
const D = (datos as any).svm_suave
const Cs = Object.keys(D.C).sort((a, b) => Number(a) - Number(b))
const sel = ref('1')
const S = computed(() => D.C[sel.value])
const dom = [-2.4, 3.6, -2.8, 2.9]
const W = 600, H = 440, m = { l: 16, r: 16, t: 16, b: 16 }
const x = scaleLinear().domain([dom[0], dom[1]]).range([m.l, W - m.r]), y = scaleLinear().domain([dom[2], dom[3]]).range([H - m.b, m.t])
const seg = (c: number) => { const s = recta(S.value.w, S.value.b, c, dom); return s ? { x1: x(s[0][0]), y1: y(s[0][1]), x2: x(s[1][0]), y2: y(s[1][1]) } : { x1: 0, y1: 0, x2: 0, y2: 0 } }
const vs = computed(() => new Set(S.value.vs as number[]))
</script>

<template>
  <div class="ms">
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" fill="#F6F7FA" rx="8" />
      <line v-bind="seg(1)" stroke="#C9971C" stroke-width="2" stroke-dasharray="4 4" />
      <line v-bind="seg(-1)" stroke="#C9971C" stroke-width="2" stroke-dasharray="4 4" />
      <line v-bind="seg(0)" stroke="#C9971C" stroke-width="3.5" />
      <circle v-for="(v, i) in D.x" :key="i" :cx="x(v)" :cy="y(D.y[i])" r="5" :fill="D.g[i] ? '#D5433C' : '#24398A'" :opacity="vs.has(i) ? 1 : 0.35" />
      <text :x="x(-2.2)" :y="y(-2.5)" style="fill: #1A2133; font-weight: 700">Adelie</text>
      <text :x="x(2.2)" :y="y(2.5)" style="fill: #1A2133; font-weight: 700">Chinstrap</text>
    </svg>
    <div class="ms__lado">
      <div class="ms__botones"><button v-for="c in Cs" :key="c" :class="{ activo: c === sel }" @click="sel = c">C = {{ c }}</button></div>
      <table>
        <tr><td>vectores de soporte</td><td><b>{{ S.n_vs }}</b> de {{ D.x.length }}</td></tr>
        <tr><td>ancho del margen 2/‖w‖</td><td><b>{{ S.ancho.toFixed(2) }}</b></td></tr>
        <tr><td>mal clasificados (entrenamiento)</td><td><b>{{ S.errores }}</b></td></tr>
        <tr><td>exactitud (validación cruzada)</td><td><b>{{ (100 * S.cv).toFixed(1) }} %</b></td></tr>
      </table>
      <p class="ms__nota">Los puntos opacos son los vectores de soporte: los que están sobre el margen, dentro de él o del lado equivocado.</p>
    </div>
  </div>
</template>

<style scoped>
.ms { display: grid; grid-template-columns: 600px 1fr; gap: 36px; align-items: start; }
.ms__lado { display: flex; flex-direction: column; gap: 18px; font-size: 20px; }
.ms__botones { display: flex; gap: 8px; flex-wrap: wrap; }
.ms__botones button { font: inherit; font-size: 18px; padding: 6px 14px; border: 1.5px solid var(--azul); border-radius: 6px; background: #fff; color: var(--azul); cursor: pointer; }
.ms__botones button.activo { background: var(--azul); color: #fff; }
.ms table td { padding: 6px 10px; border-bottom: 1px solid var(--linea); }
.ms table b { color: var(--azul); font-variant-numeric: tabular-nums; }
.ms__nota { font-size: 17px; color: var(--tenue); }
</style>
