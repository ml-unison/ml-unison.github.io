<script setup lang="ts">
// Adam sin regularizar: pérdida y exactitud en entrenamiento y validación, con la época de menor pérdida de validación.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-02.json'
const C = (datos as any).corridas as any[]
const props = withDefaults(defineProps<{ corrida?: string }>(), { corrida: 'Adam' })
const h = C.find(c => c.nombre === props.corrida)
const W = 540, H = 340, m = { l: 60, r: 130, t: 40, b: 66 }
const x = scaleLinear().domain([1, 30]).range([m.l, W - m.r])
const yp = scaleLinear().domain([0, 0.8]).range([H - m.b, m.t])
const ya = scaleLinear().domain([0.75, 1.0]).range([H - m.b, m.t])
const d = (v: number[], y: any) => line<number>().x((_, i) => x(i + 1)).y(w => y(w))(v)!
const f2 = (v: number) => v.toFixed(2)
</script>

<template>
  <div class="pt">
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <text :x="m.l" :y="18" style="fill: #20305A; font-weight: 650; font-size: 20px">entropía cruzada</text>
      <Ejes :x="x" :y="yp" :x-ticks="[1, 10, 20, 30]" :y-ticks="[0, 0.2, 0.4, 0.6, 0.8]" :y-formato="f2" x-titulo="época" />
      <line :x1="x(h.epoca)" :x2="x(h.epoca)" :y1="m.t" :y2="H - m.b" stroke="#C9971C" stroke-width="2.5" stroke-dasharray="6 4" />
      <path :d="d(h.pe, yp)" fill="none" stroke="#24398A" stroke-width="3" />
      <path :d="d(h.pv, yp)" fill="none" stroke="#D5433C" stroke-width="3" />
      <circle :cx="x(h.epoca)" :cy="yp(h.pv_min)" r="7" fill="#C9971C" stroke="#fff" stroke-width="2" />
      <text :x="W - m.r + 10" :y="yp(h.pe[29]) + 6" style="fill: #1A2133; font-weight: 600; font-size: 18px">entrenamiento</text>
      <text :x="W - m.r + 10" :y="yp(h.pv[29]) + 6" style="fill: #1A2133; font-weight: 600; font-size: 18px">validación</text>
      <text :x="x(h.epoca) + 8" :y="m.t + 14" style="fill: #A87C12; font-weight: 600; font-size: 17px">época {{ h.epoca }}</text>
    </svg>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <text :x="m.l" :y="18" style="fill: #20305A; font-weight: 650; font-size: 20px">exactitud</text>
      <Ejes :x="x" :y="ya" :x-ticks="[1, 10, 20, 30]" :y-ticks="[0.75, 0.8, 0.85, 0.9, 0.95, 1.0]" :y-formato="f2" x-titulo="época" />
      <line :x1="x(h.epoca)" :x2="x(h.epoca)" :y1="m.t" :y2="H - m.b" stroke="#C9971C" stroke-width="2.5" stroke-dasharray="6 4" />
      <path :d="d(h.ae, ya)" fill="none" stroke="#24398A" stroke-width="3" />
      <path :d="d(h.av, ya)" fill="none" stroke="#D5433C" stroke-width="3" />
      <text :x="W - m.r + 10" :y="ya(h.ae[29]) + 6" style="fill: #1A2133; font-weight: 600; font-size: 18px">entrenamiento</text>
      <text :x="W - m.r + 10" :y="ya(h.av[29]) + 6" style="fill: #1A2133; font-weight: 600; font-size: 18px">validación</text>
    </svg>
  </div>
</template>

<style scoped>
.pt { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
</style>
