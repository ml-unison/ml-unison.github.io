<script setup lang="ts">
// Gradient boosting con tocones (η = 0.5) sobre la hora, en días laborales de las bicicletas de Washington.
// Izquierda: datos y F_m. Derecha: residuos y − F_m y el tocón que se les ajusta en la ronda m + 1.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u05-03.json'
const E = (datos as any).etapas
const etapas = Object.keys(E.m).sort((a, b) => Number(a) - Number(b))
const sel = ref('1')
const S = computed(() => E.m[sel.value])
const W = 540, H = 400, m = { l: 62, r: 14, t: 40, b: 74 }
const x = scaleLinear().domain([-0.6, 23.6]).range([m.l, W - m.r])
const yA = scaleLinear().domain([0, 1000]).range([H - m.b, m.t])
const yB = scaleLinear().domain([-500, 700]).range([H - m.b, m.t])
const escalon = (v: number[], y: any) => {
  let d = `M${x(-0.5)},${y(v[0])}`
  v.forEach((f, h) => { d += `V${y(f)}H${x(h + 0.5)}` })
  return d
}
const residuos = computed(() => E.y.map((v: number, i: number) => v - S.value.F[Math.min(23, Math.max(0, Math.round(E.x[i])))]))
const ultimo = computed(() => sel.value === etapas[etapas.length - 1])
</script>

<template>
  <div class="be">
    <div class="be__botones">
      <span>tocones sumados <i>t</i></span>
      <button v-for="k in etapas" :key="k" :class="{ activo: k === sel }" @click="sel = k">{{ k }}</button>
    </div>
    <div class="be__paneles">
      <div>
        <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
          <Ejes :x="x" :y="yA" :x-ticks="[0, 6, 12, 18, 23]" :y-ticks="[0, 250, 500, 750, 1000]" x-titulo="hora (días laborales)" y-titulo="bicicletas por hora" />
          <circle v-for="(v, i) in E.x" :key="i" :cx="x(v)" :cy="yA(E.y[i])" r="2.4" fill="#6B7385" opacity="0.28" />
          <template v-if="ultimo"><circle v-for="(v, h) in E.promedio" :key="`p${h}`" :cx="x(h)" :cy="yA(v)" r="5" fill="#C9971C" /></template>
          <path :d="escalon(S.F, yA)" fill="none" stroke="#24398A" stroke-width="3.5" />
        </svg>
        <div class="be__t"><i>F</i><sub>t</sub> = promedio + η · (suma de {{ sel }} {{ sel === '1' ? 'tocón' : 'tocones' }})<template v-if="ultimo"> · <span style="color: #A87C12">● promedio por hora</span></template></div>
      </div>
      <div>
        <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
          <Ejes :x="x" :y="yB" :x-ticks="[0, 6, 12, 18, 23]" :y-ticks="[-500, -250, 0, 250, 500]" x-titulo="hora (días laborales)" y-titulo="residuo y − Fₜ" />
          <line :x1="m.l" :x2="W - m.r" :y1="yB(0)" :y2="yB(0)" stroke="#5E6678" stroke-width="1.2" />
          <circle v-for="(v, i) in E.x" :key="i" :cx="x(v)" :cy="yB(residuos[i])" r="2.4" fill="#6B7385" opacity="0.28" />
          <path v-if="!ultimo" :d="escalon(S.h, yB)" fill="none" stroke="#D5433C" stroke-width="3.5" />
        </svg>
        <div class="be__t">residuos (pseudo-residuos) <span v-if="!ultimo" style="color: #D5433C">y el tocón <i>h</i><sub>t+1</sub> que se les ajusta</span></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.be { display: flex; flex-direction: column; gap: 6px; }
.be__botones { display: flex; gap: 8px; align-items: center; font-size: 20px; }
.be__botones span { color: var(--tenue); margin-right: 6px; }
.be__botones button { font: inherit; font-size: 18px; min-width: 48px; padding: 4px 10px; border: 1.5px solid var(--azul); border-radius: 6px; background: #fff; color: var(--azul); cursor: pointer; }
.be__botones button.activo { background: var(--azul); color: #fff; }
.be__paneles { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
.be__t { text-align: center; font-size: 18px; color: var(--marino); margin-top: -2px; }
</style>
