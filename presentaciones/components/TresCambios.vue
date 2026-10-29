<script setup lang="ts">
// Las tres maneras de cambiar, con las bicicletas: P(x) (temperatura de marzo), P(y|x) (rentas según la temperatura)
// y P(y) (rentas de junio). Azul = 2011, rojo = 2012.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import d from '../datos/u08-02.json'
const C = (d as any).cambios
const props = withDefaults(defineProps<{ mostrar?: number }>(), { mostrar: 3 })
const W = 360, H = 290, m = { l: 52, r: 12, t: 20, b: 52 }
const escalon = (bordes: number[], v: number[], x: any, y: any) => {
  let s = `M${x(bordes[0])},${y(0)}`
  v.forEach((h, i) => { s += `L${x(bordes[i])},${y(h)}L${x(bordes[i + 1])},${y(h)}` })
  return s + `L${x(bordes[bordes.length - 1])},${y(0)}`
}
const xa = scaleLinear().domain([0, 34]).range([m.l, W - m.r]), ya = scaleLinear().domain([0, 0.34]).range([H - m.b, m.t])
const xb = scaleLinear().domain([0, 40]).range([m.l, W - m.r]), yb = scaleLinear().domain([0, 420]).range([H - m.b, m.t])
const xc = scaleLinear().domain([0, 950]).range([m.l, W - m.r]), yc = scaleLinear().domain([0, 0.24]).range([H - m.b, m.t])
const curva = (k: string) => line<number>().x((_, i) => xb(C.curva[k].x[i])).y(v => yb(v))(C.curva[k].y)!
const paneles = [
  { t: 'ℙ(<b>x</b>) · temperatura de marzo', n: 1 },
  { t: 'ℙ(y | <b>x</b>) · rentas según la temperatura', n: 2 },
  { t: 'ℙ(y) · rentas por hora en junio', n: 3 },
]
</script>

<template>
  <div class="tc">
    <div v-for="p in paneles" :key="p.n" :style="{ opacity: mostrar >= p.n ? 1 : 0, transition: 'opacity .4s' }">
      <div class="tc__t" v-html="p.t"></div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <template v-if="p.n === 1">
          <Ejes :x="xa" :y="ya" :x-ticks="[0, 10, 20, 30]" :y-ticks="[0, 0.1, 0.2, 0.3]" x-titulo="°C" />
          <path :d="escalon(C.temp.bordes, C.temp.a, xa, ya)" fill="#24398A" fill-opacity="0.12" stroke="#24398A" stroke-width="2.6" />
          <path :d="escalon(C.temp.bordes, C.temp.b, xa, ya)" fill="#D5433C" fill-opacity="0.12" stroke="#D5433C" stroke-width="2.6" />
          <text :x="xa(3)" :y="ya(0.27)" class="etiqueta" style="fill:#24398A">2011</text>
          <text :x="xa(24)" :y="ya(0.17)" class="etiqueta" style="fill:#D5433C">2012</text>
        </template>
        <template v-else-if="p.n === 2">
          <Ejes :x="xb" :y="yb" :x-ticks="[0, 10, 20, 30, 40]" :y-ticks="[0, 100, 200, 300, 400]" x-titulo="°C" />
          <path :d="curva('2011')" fill="none" stroke="#24398A" stroke-width="3" />
          <path :d="curva('2012')" fill="none" stroke="#D5433C" stroke-width="3" />
          <text :x="xb(29)" :y="yb(170)" class="etiqueta" style="fill:#24398A">2011</text>
          <text :x="xb(14)" :y="yb(250)" class="etiqueta" style="fill:#D5433C">2012</text>
        </template>
        <template v-else>
          <Ejes :x="xc" :y="yc" :x-ticks="[0, 300, 600, 900]" :y-ticks="[0, 0.1, 0.2]" x-titulo="bicicletas en la hora" />
          <path :d="escalon(C.rentas.bordes, C.rentas.a, xc, yc)" fill="#24398A" fill-opacity="0.12" stroke="#24398A" stroke-width="2.6" />
          <path :d="escalon(C.rentas.bordes, C.rentas.b, xc, yc)" fill="#D5433C" fill-opacity="0.12" stroke="#D5433C" stroke-width="2.6" />
          <text :x="xc(240)" :y="yc(0.155)" class="etiqueta" style="fill:#24398A">2011</text>
          <text :x="xc(600)" :y="yc(0.07)" class="etiqueta" style="fill:#D5433C">2012</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.tc { display: grid; grid-template-columns: repeat(3, 1fr); gap: 26px; }
.tc__t { font-weight: 650; font-size: 20px; color: var(--marino); margin-bottom: 4px; }
</style>
