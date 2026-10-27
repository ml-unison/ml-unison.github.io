<script setup lang="ts">
// Del pronóstico a la regresión: desde el origen t (domingo 16 de sept. de 2012, 20 h) se arma el renglón de
// atributos para el horizonte h. Un deslizador mueve h.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-03.json'
const V = (datos as any).ventana
const Y = V.y as number[], o = V.origen as number
const h = ref(12)
const W = 1120, H = 360, m = { l: 66, r: 20, t: 44, b: 40 }
const x = scaleLinear().domain([0, Y.length - 1]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 1000]).range([H - m.b, m.t])
const DIA = ['dom', 'lun', 'mar', 'mié', 'jue', 'vie', 'sáb']
const i0 = new Date(V.inicio + 'T00:00:00')
const dias = Array.from({ length: Y.length / 24 }, (_, k) => { const f = new Date(i0.getTime() + k * 864e5); return { k, t: `${DIA[f.getDay()]} ${f.getDate()}` } })
const pasado = line<number>().x((_, i) => x(i)).y(v => y(v))(Y.slice(0, o + 1))!
const futuro = line<number>().x((_, i) => x(o + i)).y(v => y(v))(Y.slice(o))!
const rezagos = Array.from({ length: 24 }, (_, k) => o - k)
const obj = computed(() => o + h.value)
const ayer = computed(() => obj.value - 24)
const semana = computed(() => obj.value - 168)
const horaObj = computed(() => { const f = new Date(i0.getTime() + obj.value * 36e5); return `${DIA[f.getDay()]} ${f.getDate()}, ${f.getHours()} h` })
</script>

<template>
  <div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <Ejes :x="x" :y="y" :x-ticks="[]" :y-ticks="[0, 250, 500, 750, 1000]" y-titulo="rentas por hora" />
      <line v-for="d in dias" :key="`l${d.k}`" :x1="x(d.k * 24)" :x2="x(d.k * 24)" :y1="m.t" :y2="H - m.b" stroke="#E2E5EC" />
      <text v-for="d in dias" :key="d.k" :x="x(d.k * 24 + 12)" :y="H - m.b + 26" text-anchor="middle">{{ d.t }}</text>
      <rect :x="x(o - 23)" :y="m.t" :width="x(o) - x(o - 23)" :height="H - m.t - m.b" fill="#C0CEE3" opacity="0.5" />
      <path :d="pasado" fill="none" stroke="#6B7385" stroke-width="1.6" />
      <path :d="futuro" fill="none" stroke="#8A91A0" stroke-width="1.4" stroke-dasharray="4 4" />
      <circle v-for="i in rezagos" :key="i" :cx="x(i)" :cy="y(Y[i])" r="4" fill="#24398A" />
      <line :x1="x(o)" :x2="x(o)" :y1="m.t - 10" :y2="H - m.b" stroke="#1A2133" stroke-width="2" />
      <text :x="x(o) - 8" :y="m.t - 16" text-anchor="end" style="fill: #1A2133; font-weight: 600">origen t</text>
      <text :x="x(o) + 8" :y="m.t - 16" style="fill: #5E6678">futuro: aún no se conoce</text>
      <circle :cx="x(semana)" :cy="y(Y[semana])" r="9" fill="#C9971C" stroke="#fff" stroke-width="2" />
      <circle :cx="x(ayer)" :cy="y(Y[ayer])" r="9" fill="#C9971C" stroke="#fff" stroke-width="2" />
      <text :x="x(semana)" :y="y(Y[semana]) - 16" text-anchor="middle" style="fill: #1A2133; font-weight: 600">y(t+h−168)</text>
      <text :x="x(ayer)" :y="y(Y[ayer]) - 16" text-anchor="middle" style="fill: #1A2133; font-weight: 600">y(t+h−24)</text>
      <circle :cx="x(obj)" :cy="y(Y[obj])" r="10" fill="#D5433C" stroke="#fff" stroke-width="2" />
      <text :x="x(obj) + 14" :y="y(Y[obj]) - 12" style="fill: #D5433C; font-weight: 700">y(t+h)</text>
    </svg>
    <div class="deslizador">
      <span style="white-space: nowrap">h =</span>
      <input type="range" min="1" max="24" step="1" v-model.number="h" />
      <output>{{ h }}</output>
      <span class="tenue" style="font-size: 20px; white-space: nowrap">objetivo: <b style="color: #D5433C">{{ horaObj }}</b> · <span style="color: #24398A">●</span> últimas 24 h · <span style="color: #A87C12">●</span> misma hora ayer y hace una semana</span>
    </div>
  </div>
</template>
