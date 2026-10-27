<script setup lang="ts">
// Caminatas aleatorias simuladas con la envolvente ±1.96√t, y la varianza de 2,000 caminatas contra tσ².
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-01.json'
const props = withDefaults(defineProps<{ varianza?: boolean }>(), { varianza: true })
const C = (datos as any).caminata
const W = 540, H = 380, m = { l: 62, r: 20, t: 40, b: 64 }
const x = scaleLinear().domain([0, 500]).range([m.l, W - m.r])
const y = scaleLinear().domain([-60, 60]).range([H - m.b, m.t])
const yv = scaleLinear().domain([0, 550]).range([H - m.b, m.t])
const caminos = C.caminos.map((c: number[]) => line<number>().x((_, i) => x(C.t[i])).y(v => y(v))(c)!)
const ts = Array.from({ length: 101 }, (_, i) => i * 5)
const sup = line<number>().x(t => x(t)).y(t => y(1.96 * Math.sqrt(t)))(ts)!
const inf = line<number>().x(t => x(t)).y(t => y(-1.96 * Math.sqrt(t)))(ts)!
const vEmp = line<number>().x((_, i) => x(C.t[i])).y(v => yv(v))(C.var)!
const vTeo = line<number>().x(t => x(t)).y(t => yv(t))(ts)!
</script>

<template>
  <div class="cam" :style="{ gridTemplateColumns: varianza ? '1fr 1fr' : '1fr' }">
    <div>
      <div class="cam__t">25 caminatas, yₜ = yₜ₋₁ + εₜ</div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
        <Ejes :x="x" :y="y" :x-ticks="[0, 100, 200, 300, 400, 500]" :y-ticks="[-50, 0, 50]" x-titulo="t" />
        <path v-for="(c, k) in caminos" :key="k" :d="c" fill="none" stroke="#24398A" stroke-width="1.2" opacity="0.5" />
        <path :d="sup" fill="none" stroke="#D5433C" stroke-width="3.5" />
        <path :d="inf" fill="none" stroke="#D5433C" stroke-width="3.5" />
        <text :x="x(495)" :y="y(1.96 * Math.sqrt(495)) - 12" text-anchor="end" style="fill: #1A2133; font-weight: 600">±1.96√t</text>
      </svg>
    </div>
    <div v-if="varianza">
      <div class="cam__t">Varianza de 2,000 caminatas</div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
        <Ejes :x="x" :y="yv" :x-ticks="[0, 100, 200, 300, 400, 500]" :y-ticks="[0, 100, 200, 300, 400, 500]" x-titulo="t" />
        <path :d="vTeo" fill="none" stroke="#D5433C" stroke-width="3" stroke-dasharray="9 7" />
        <path :d="vEmp" fill="none" stroke="#24398A" stroke-width="3.5" />
        <text :x="x(150)" :y="yv(330)" style="fill: #1A2133; font-weight: 600">empírica</text>
        <text :x="x(330)" :y="yv(200)" style="fill: #1A2133; font-weight: 600">teórica, tσ²</text>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.cam { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
.cam__t { font-weight: 650; font-size: 21px; color: var(--marino); margin-bottom: 2px; }
</style>
