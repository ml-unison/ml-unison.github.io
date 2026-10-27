<script setup lang="ts">
// AdaBoost con tocones en los pingüinos (barbijo contra las otras dos especies): regiones de signo(F_t),
// el tocón recién agregado y el peso de cada punto para la ronda siguiente (área ∝ peso).
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u05-03.json'
const A = (datos as any).adaboost
const rondas = Object.keys(A.rondas).sort((a, b) => Number(a) - Number(b))
const sel = ref(rondas[0])
const R = computed(() => A.rondas[sel.value])
const W = 640, H = 440, m = { l: 68, r: 14, t: 14, b: 58 }
const x = scaleLinear().domain([31, 61]).range([m.l, W - m.r])
const y = scaleLinear().domain([12.5, 22]).range([H - m.b, m.t])
const radio = (w: number) => Math.min(15, Math.max(0.9, 2.7 * Math.sqrt(w)))
const nombres = ['longitud', 'profundidad']
const orden = computed(() => A.x.map((_: number, i: number) => i).sort((i: number, j: number) => R.value.w[j] - R.value.w[i]))
</script>

<template>
  <div class="ap">
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <clipPath id="ap-clip"><rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" /></clipPath>
      <g clip-path="url(#ap-clip)">
        <Regiones :key="sel" :datos="R.regiones" :x="x" :y="y" :colores="['#C0CEE3', '#F3C7C4']" :opacidad="0.75" />
      </g>
      <Ejes :x="x" :y="y" :x-ticks="[35, 40, 45, 50, 55, 60]" :y-ticks="[14, 16, 18, 20, 22]" x-titulo="longitud del pico (mm)" />
      <g clip-path="url(#ap-clip)">
        <circle v-for="i in orden" :key="i" :cx="x(A.x[i])" :cy="y(A.y[i])" :r="radio(R.w[i])"
                :fill="A.g[i] ? '#D5433C' : '#24398A'" fill-opacity="0.85" stroke="#fff" stroke-width="0.8" />
        <line v-if="R.j === 0" :x1="x(R.u)" :x2="x(R.u)" :y1="m.t" :y2="H - m.b" stroke="#1A2133" stroke-width="2.2" stroke-dasharray="7 5" />
        <line v-else :x1="m.l" :x2="W - m.r" :y1="y(R.u)" :y2="y(R.u)" stroke="#1A2133" stroke-width="2.2" stroke-dasharray="7 5" />
      </g>
      <text :x="16" :y="(H - m.b + m.t) / 2" class="titulo-eje" text-anchor="middle" :transform="`rotate(-90 16 ${(H - m.b + m.t) / 2})`">profundidad del pico (mm)</text>
    </svg>
    <div class="ap__lado">
      <div class="ap__botones">
        <span>ronda</span>
        <button v-for="t in rondas" :key="t" :class="{ activo: t === sel }" @click="sel = t">{{ t }}</button>
      </div>
      <table>
        <tr><td>tocón <i>h</i><sub>t</sub></td><td><b>barbijo</b> si {{ nombres[R.j] }} &gt; {{ R.u.toFixed(1) }} mm</td></tr>
        <tr><td>error ponderado <i>ε</i><sub>t</sub></td><td><b>{{ R.eps.toFixed(3) }}</b></td></tr>
        <tr><td>voto <i>α</i><sub>t</sub></td><td><b>{{ R.alfa.toFixed(3) }}</b></td></tr>
        <tr><td>error de signo(<i>F</i><sub>t</sub>)</td><td><b>{{ R.error.toFixed(3) }}</b></td></tr>
      </table>
      <p class="ap__nota">
        <span class="ap__punto" style="background: #D5433C" /> barbijo
        <span class="ap__punto" style="background: #24398A; margin-left: 14px" /> Adelia y papúa<br>
        El área de cada punto es su peso en la ronda siguiente; la línea punteada es el último tocón.
      </p>
    </div>
  </div>
</template>

<style scoped>
.ap { display: grid; grid-template-columns: 640px 1fr; gap: 34px; align-items: start; }
.ap__lado { display: flex; flex-direction: column; gap: 18px; font-size: 20px; padding-top: 6px; }
.ap__botones { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.ap__botones span { color: var(--tenue); margin-right: 4px; }
.ap__botones button { font: inherit; font-size: 18px; min-width: 50px; padding: 5px 10px; border: 1.5px solid var(--azul); border-radius: 6px; background: #fff; color: var(--azul); cursor: pointer; }
.ap__botones button.activo { background: var(--azul); color: #fff; }
.ap table { font-size: 19px; }
.ap table td { padding: 7px 10px 7px 0; border-bottom: 1px solid var(--linea); }
.ap table td:first-child { font-weight: 400; color: var(--tenue); }
.ap table b { color: var(--azul); font-variant-numeric: tabular-nums; }
.ap__nota { font-size: 17px; color: var(--tenue); line-height: 1.4; }
.ap__punto { display: inline-block; width: 12px; height: 12px; border-radius: 50%; vertical-align: -1px; margin-right: 4px; }
</style>
