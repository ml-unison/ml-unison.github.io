<script setup lang="ts">
// Inestabilidad: árboles sin límite de profundidad entrenados en cuatro remuestras bootstrap de los
// pingüinos, y la votación de 200 de ellos.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { SERIE } from './colores'
import datos from '../datos/u05-01.json'
const I = (datos as any).inestabilidad, P = (datos as any).pinguinos
const sel = ref(0)
const W = 620, H = 440, m = { l: 64, r: 12, t: 24, b: 64 }
const x = scaleLinear().domain([31, 61]).range([m.l, W - m.r]), y = scaleLinear().domain([12.5, 22]).range([H - m.b, m.t])
const TINTAS = ['#C0CEE3', '#F3C7C4', '#EFDDAA']
const R = computed(() => (sel.value < 4 ? I.remuestras[sel.value] : null))
const opciones = ['remuestra 1', 'remuestra 2', 'remuestra 3', 'remuestra 4', 'votación de 200']
</script>

<template>
  <div class="ra">
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <clipPath id="ra-clip"><rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" /></clipPath>
      <g v-if="R">
        <rect v-for="(q, k) in R.rect" :key="`${sel}-${k}`" :x="x(q[0])" :y="y(q[3])" :width="x(q[1]) - x(q[0])" :height="y(q[2]) - y(q[3])"
              :fill="TINTAS[q[4]]" stroke="#fff" stroke-width="1.2" />
      </g>
      <g v-else :clip-path="'url(#ra-clip)'"><Regiones :datos="I.voto" :x="x" :y="y" :opacidad="1" /></g>
      <Ejes :x="x" :y="y" :x-ticks="[35, 40, 45, 50, 55, 60]" :y-ticks="[14, 16, 18, 20, 22]" x-titulo="largo del pico (mm)" y-titulo="profundidad del pico (mm)" />
      <circle v-for="(v, i) in P.x" :key="i" :cx="x(v)" :cy="y(P.y[i])" r="3.2" :fill="SERIE[P.g[i]]" stroke="#fff" stroke-width="0.6" opacity="0.75" />
    </svg>
    <div class="ra__lado">
      <div class="ra__botones"><button v-for="(o, k) in opciones" :key="o" :class="{ activo: k === sel }" @click="sel = k">{{ o }}</button></div>
      <p v-if="R">{{ R.hojas }} hojas. Primera pregunta: ¿{{ R.raiz }} ≤ {{ R.umbral.toFixed(2) }}?</p>
      <p v-else>Cada punto del plano se asigna a la especie que eligen más árboles. La frontera se estabiliza.</p>
      <slot />
    </div>
  </div>
</template>

<style scoped>
.ra { display: grid; grid-template-columns: 620px 1fr; gap: 36px; align-items: start; }
.ra__lado { display: flex; flex-direction: column; gap: 14px; font-size: 21px; }
.ra__lado p { margin: 0; }
.ra__botones { display: flex; gap: 8px; flex-wrap: wrap; }
.ra__botones button { font: inherit; font-size: 17px; padding: 5px 12px; border: 1.5px solid var(--azul); border-radius: 6px; background: #fff; color: var(--azul); cursor: pointer; }
.ra__botones button.activo { background: var(--azul); color: #fff; }
</style>
