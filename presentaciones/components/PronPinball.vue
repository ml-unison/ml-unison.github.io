<script setup lang="ts">
// Pérdida cuantil: a la izquierda ρ_τ(u); a la derecha, la pérdida media de un pronóstico constante q para las rentas
// de las 8 h en días laborales, con el histograma de los datos. El mínimo cae en el cuantil τ.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-03.json'
const P = (datos as any).pinball
const k = ref(1) // τ = 0.1
const tau = computed(() => P.tau[k.value] as number)
// panel izquierdo
const Wl = 400, Hl = 330, ml = { l: 40, r: 20, t: 30, b: 60 }
const xl = scaleLinear().domain([-1, 1]).range([ml.l, Wl - ml.r]), yl = scaleLinear().domain([0, 1]).range([Hl - ml.b, ml.t])
const ve = computed(() => line<number>().x(u => xl(u)).y(u => yl(u * (tau.value - (u < 0 ? 1 : 0))))([-1, 0, 1])!)
// panel derecho
const W = 660, H = 360, m = { l: 60, r: 20, t: 30, b: 60 }
const x = scaleLinear().domain([0, 1000]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 420]).range([H - m.b, m.t])
const maxH = Math.max(...P.hist)
const yh = scaleLinear().domain([0, maxH]).range([H - m.b, H - m.b - 150])
const curva = computed(() => line<number>().x((_, i) => x(P.q[i])).y(v => y(v)).defined(v => v <= 440)(P.curvas[k.value])!)
const debajo = computed(() => Math.round(100 * tau.value))
</script>

<template>
  <div>
    <div class="pb">
      <svg class="grafica" :viewBox="`0 0 ${Wl} ${Hl}`">
        <Ejes :x="xl" :y="yl" :x-ticks="[-1, 0, 1]" :y-ticks="[0, 0.5, 1]" x-titulo="residuo u = y − q" />
        <path :d="ve" fill="none" stroke="#24398A" stroke-width="3.5" />
        <text :x="xl(-0.5)" :y="ml.t - 6" text-anchor="middle" style="fill: #1A2133; font-size: 17px">pasarse: pendiente {{ (1 - tau).toFixed(2) }}</text>
        <text :x="xl(0.5)" :y="ml.t + 18" text-anchor="middle" style="fill: #1A2133; font-size: 17px">corto: {{ tau.toFixed(2) }}</text>
      </svg>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <Ejes :x="x" :y="y" :x-ticks="[0, 200, 400, 600, 800, 1000]" :y-ticks="[0, 100, 200, 300, 400]" x-titulo="pronóstico constante q (rentas a las 8 h, días laborales)" y-titulo="pérdida cuantil media" />
        <rect v-for="(c, i) in P.hist" :key="i" :x="x(i * P.ancho_hist) + 1" :y="yh(c)" :width="x(P.ancho_hist) - x(0) - 2" :height="yh(0) - yh(c)"
              :fill="(i + 0.5) * P.ancho_hist < P.minimo[k] ? '#F3C7C4' : '#C0CEE3'" />
        <path :d="curva" fill="none" stroke="#1A2133" stroke-width="3" />
        <line :x1="x(P.minimo[k])" :x2="x(P.minimo[k])" :y1="m.t" :y2="H - m.b" stroke="#D5433C" stroke-width="2" stroke-dasharray="6 4" />
        <circle :cx="x(P.minimo[k])" :cy="y(P.valor_min[k])" r="8" fill="#D5433C" stroke="#fff" stroke-width="2" />
        <text :x="x(P.minimo[k]) + (P.minimo[k] > 700 ? -10 : 10)" :y="m.t + 16" :text-anchor="P.minimo[k] > 700 ? 'end' : 'start'" style="fill: #1A2133; font-weight: 600">mínimo en q = {{ P.minimo[k] }}</text>
        <text :x="x(P.minimo[k]) + (P.minimo[k] > 700 ? -10 : 10)" :y="m.t + 40" :text-anchor="P.minimo[k] > 700 ? 'end' : 'start'" style="fill: #5E6678; font-size: 17px">cuantil {{ tau.toFixed(2) }} de los datos: {{ P.cuantil[k] }}</text>
      </svg>
    </div>
    <div class="deslizador">
      <span>τ</span>
      <input type="range" min="0" :max="P.tau.length - 1" step="1" v-model.number="k" />
      <output>{{ tau.toFixed(2) }}</output>
      <span class="tenue" style="font-size: 20px">{{ debajo }} % de los {{ P.n }} días quedan a la izquierda del mínimo (rojo)</span>
    </div>
  </div>
</template>

<style scoped>
.pb { display: grid; grid-template-columns: 400px 660px; gap: 40px; align-items: end; }
</style>
