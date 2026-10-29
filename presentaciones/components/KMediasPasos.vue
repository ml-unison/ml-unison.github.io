<script setup lang="ts">
// El algoritmo de Lloyd paso a paso con clientes reales (log frecuencia vs log monto, estandarizados).
import { computed, ref } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u02.json'
import { SERIE } from './colores'
const L = (datos as any).lloyd
const n = L.pasos.length
const paso = ref(0)
const P = computed(() => L.pasos[paso.value])
// En el paso 0 los puntos aún no tienen grupo; luego mostramos asignar (a) y mover (b)
const W = 360, H = 220, m = { l: 52, r: 12, t: 44, b: 34 }
const xj = scaleLinear().domain([0, n - 1]).range([m.l, W - m.r])
const yj = scaleLinear().domain([0, Math.max(...L.pasos.map((p: any) => p.J)) * 1.05]).range([H - m.b, m.t])
const camino = computed(() => line<any>().x((_, i) => xj(i)).y((p: any) => yj(p.J))(L.pasos.slice(0, paso.value + 1))!)
const rastro = (j: number) => L.pasos.slice(0, paso.value + 1).map((p: any) => p.C[j])
</script>

<template>
  <div class="kp">
    <div>
      <Dispersion :x="L.x" :y="L.y" :g="paso === 0 ? L.x.map(() => -1) : P.e" :centros="P.C" :ancho="640" :alto="440" :radio="4.2" :opacidad="0.75">
        <template #default="{ x, y }">
          <polyline v-for="j in 4" :key="j" :points="rastro(j - 1).map((c: number[]) => `${x(c[0])},${y(c[1])}`).join(' ')" fill="none" stroke="#1A2133" stroke-width="1.6" stroke-dasharray="4 4" opacity="0.6" />
        </template>
      </Dispersion>
      <p class="kp__ejes">horizontal: log(número de compras) · vertical: log(monto gastado), ambos estandarizados</p>
    </div>
    <div class="kp__lado">
      <div class="deslizador"><span>Iteración</span><input v-model.number="paso" type="range" min="0" :max="n - 1" step="1" /><b>{{ paso }}</b></div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <g class="eje"><line :x1="m.l" :x2="W - m.r" :y1="H - m.b" :y2="H - m.b" />
          <text v-for="t in [0, 10, 20]" :key="t" :x="xj(t)" :y="H - m.b + 24" text-anchor="middle">{{ t }}</text>
          <text v-for="t in yj.ticks(3)" :key="`y${t}`" :x="m.l - 8" :y="yj(t) + 6" text-anchor="end">{{ t }}</text>
        </g>
        <path :d="camino" fill="none" :stroke="SERIE[0]" stroke-width="3" />
        <circle :cx="xj(paso)" :cy="yj(P.J)" r="6" :fill="SERIE[0]" stroke="#fff" stroke-width="2" />
        <text :x="m.l - 40" :y="18" class="titulo-eje" style="font-size: 17px">J (suma de distancias²)</text>
      </svg>
      <div class="kp__j">J = <b>{{ P.J.toFixed(1) }}</b></div>
      <p class="kp__nota">400 clientes de la tienda en línea, K = 4 grupos, con un inicio malo a propósito: tres centroides amontonados abajo.</p>
    </div>
  </div>
</template>

<style scoped>
.kp { display: grid; grid-template-columns: 640px 1fr; gap: 40px; align-items: start; }
.kp__ejes { font-size: 16px; color: var(--tenue); margin-top: 4px; }
.kp__lado { display: flex; flex-direction: column; gap: 12px; }
.kp__lado .deslizador input[type=range] { flex: 1 1 auto; }
.kp__j { font-size: 26px; } .kp__j b { color: var(--azul); }
.kp__nota { font-size: 18px; color: var(--tenue); }
</style>
