<script setup lang="ts">
// El grado del polinomio controla el ajuste (izquierda) y su posición en la curva de error (derecha).
import { computed, ref } from 'vue'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u01.json'
import { AZUL, DORADO } from './colores'

const P = datos.polinomios
const grado = ref(3)
const W = 520, H = 420, m = { l: 66, r: 70, t: 40, b: 70 }
const x = scaleLinear().domain([0, 14]).range([m.l, W - m.r])
const y = scaleLog().domain([1e-3, 10]).range([H - m.b, m.t]).clamp(true)
const grados = P.ein.map((_, g) => g)
const piso = (v: number) => Math.min(Math.max(v, 1e-3), 10)
const camino = (vs: number[]) => line<number>().x((_, g) => x(g)).y(v => y(piso(v)))(vs)!
const series = [
  { nombre: 'E', sub: 'out', vals: P.eout, color: AZUL, desc: 'prueba' },
  { nombre: 'E', sub: 'in', vals: P.ein, color: DORADO, desc: 'entrenamiento' },
]
const fmt = (v: number) => v > 10 ? '> 10' : v < 0.001 ? '< 0.001' : v.toPrecision(2)
const actual = computed(() => ({ ein: P.ein[grado.value], eout: P.eout[grado.value] }))
const diagnostico = computed(() => grado.value <= 1 ? 'subajuste' : grado.value >= 10 ? 'sobreajuste' : 'zona razonable')
</script>

<template>
  <div class="interactivo">
    <div>
      <label class="deslizador">
        Grado del polinomio
        <input v-model.number="grado" type="range" min="0" max="14" step="1">
        <output>{{ grado }}</output>
      </label>
      <PanelPolinomio :grado="grado" :ancho="620" :alto="350" con-real />
      <div class="lectura">
        <span>E<sub>in</sub> = <b>{{ fmt(actual.ein) }}</b></span>
        <span>E<sub>out</sub> = <b>{{ fmt(actual.eout) }}</b></span>
        <span class="diag" :class="{ mal: diagnostico !== 'zona razonable' }">{{ diagnostico }}</span>
      </div>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Error de entrenamiento y de prueba según el grado">
      <Ejes :x="x" :y="y" :x-ticks="[0, 2, 4, 6, 8, 10, 12, 14]" :y-ticks="[0.001, 0.01, 0.1, 1, 10]"
            :y-formato="(v) => String(v)" x-titulo="grado" y-titulo="error cuadrático medio (escala log)" />
      <rect :x="x(grado) - 14" :y="m.t" width="28" :height="H - m.t - m.b" fill="#EFF2F9" />
      <g v-for="s in series" :key="s.sub">
        <path :d="camino(s.vals)" fill="none" :stroke="s.color" stroke-width="3" stroke-linejoin="round" />
        <circle v-for="g in grados" :key="g" :cx="x(g)" :cy="y(piso(s.vals[g]))" :r="g === grado ? 8 : 5"
                :fill="g === grado ? s.color : '#fff'" :stroke="s.color" stroke-width="2.5" />
        <g :transform="`translate(${x(14) + 14},${y(piso(s.vals[14])) + 6})`">
          <text class="etiqueta" style="fill:#1A2133">E</text>
          <text x="13" y="6" style="fill:#1A2133; font-size: 14px; font-weight: 600">{{ s.sub }}</text>
        </g>
      </g>
      <!-- zonas grandes para elegir el grado con el ratón -->
      <rect v-for="g in grados" :key="`h${g}`" :x="x(g) - 15" :y="m.t" width="30" :height="H - m.t - m.b"
            fill="transparent" style="cursor:pointer" @click="grado = g" @mouseenter="grado = g" />
    </svg>
  </div>
</template>

<style scoped>
.interactivo { display: grid; grid-template-columns: 620px 1fr; gap: 36px; align-items: end; }
.lectura { display: flex; gap: 28px; align-items: center; font-size: 21px; color: var(--tenue); padding-left: 46px; margin-top: 4px; }
.lectura b { color: var(--tinta); font-variant-numeric: tabular-nums; }
.diag { font-family: var(--f-titulo); font-weight: 600; font-size: 13px; letter-spacing: .07em; text-transform: uppercase; padding: 4px 10px; border-radius: 20px; background: #EEF6EA; color: #3F7F2E; }
.diag.mal { background: var(--tinte-rojo); color: #B23028; }
</style>
