<script setup lang="ts">
// Temperatura media diaria de Hermosillo por día del año (promedio 1992–2024) y la red de H neuronas ReLU
// que la interpola linealmente en H + 1 nodos. Deslizador para H. Datos: herramientas/laminas/u06_01.py.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-01.json'
const props = withDefaults(defineProps<{ inicial?: number }>(), { inicial: 2 })
const D = (datos as any).hermosillo
const Hs = Object.keys(D.redes).map(Number).sort((a, b) => a - b)
const k = ref(props.inicial)
const R = computed(() => D.redes[String(Hs[k.value])])
const W = 1040, H = 350, m = { l: 56, r: 20, t: 34, b: 44 }
const x = scaleLinear().domain([1, 365]).range([m.l, W - m.r]), y = scaleLinear().domain([14, 36]).range([H - m.b, m.t])
const camino = computed(() => line<number>().x(v => x(v)).y((_, i) => y(R.value.v[i]))(R.value.t)!)
const meses = [[1, '1 ene'], [91, '1 abr'], [182, '1 jul'], [274, '1 oct'], [365, '31 dic']] as const
</script>

<template>
  <div>
    <div class="deslizador">
      <span>neuronas ocultas</span>
      <input type="range" min="0" :max="Hs.length - 1" step="1" v-model.number="k" />
      <output>{{ Hs[k] }}</output>
      <span class="tenue" style="margin-left: 18px">error cuadrático medio: <b style="color: var(--azul)">{{ R.recm.toFixed(2) }} °C</b></span>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <g class="rejilla"><line v-for="t in [15, 20, 25, 30, 35]" :key="t" :x1="m.l" :x2="W - m.r" :y1="y(t)" :y2="y(t)" /></g>
      <g class="eje">
        <line :x1="m.l" :x2="W - m.r" :y1="H - m.b" :y2="H - m.b" />
        <text v-for="[d, n] in meses" :key="d" :x="x(d)" :y="H - m.b + 26" text-anchor="middle">{{ n }}</text>
        <text v-for="t in [15, 20, 25, 30, 35]" :key="`y${t}`" :x="m.l - 10" :y="y(t) + 6" text-anchor="end">{{ t }}</text>
        <text class="titulo-eje" :x="m.l" :y="m.t - 14">temperatura media (°C)</text>
      </g>
      <circle v-for="(d, i) in D.dia" :key="d" :cx="x(d)" :cy="y(D.t[i])" r="2.4" fill="#8A91A0" />
      <path :d="camino" fill="none" stroke="#24398A" stroke-width="3.5" stroke-linejoin="round" />
      <circle v-for="(t, i) in R.t" :key="`n${i}`" :cx="x(t)" :cy="y(R.v[i])" r="5.5" fill="#C9971C" stroke="#fff" stroke-width="1.2" />
      <text :x="x(205)" :y="y(35.6)" style="fill: #6B7385; font-size: 17px">promedio de cada día, {{ D.anios[0] }}–{{ D.anios[1] }}</text>
    </svg>
  </div>
</template>
