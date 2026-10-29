<script setup lang="ts">
// El mismo código con diez semillas: desviación del AUC respecto al promedio (milésimas).
// Fila 1: bosque aleatorio, cambia la semilla del modelo. Fila 2: logística, cambia la semilla de los pliegues.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u08-01.json'
const S = (datos as any).semillas
const props = withDefaults(defineProps<{ filas?: number, referencia?: boolean }>(), { filas: 2, referencia: false })
const W = 1100, H = 310, m = { l: 330, r: 30, t: 30, b: 76 }
const x = scaleLinear().domain([-2, 2]).range([m.l, W - m.r])
const y = scaleLinear().domain([-0.5, 1.5]).range([m.t, H - m.b])
const yEje = scaleLinear().domain([0, 1]).range([H - m.b, m.t])
const desvios = (a: number[]) => { const mu = a.reduce((s, v) => s + v, 0) / a.length; return a.map(v => 1000 * (v - mu)) }
// desplazamiento vertical fijo (determinista) para que no se encimen
const jit = [-0.16, 0.1, -0.05, 0.16, -0.12, 0.04, 0.13, -0.18, 0.0, 0.08]
const series = [
  { nombre: 'Bosque aleatorio', detalle: 'cambia la semilla del modelo', color: '#D5433C', v: desvios(S.bosque), media: S.bosque.reduce((s: number, v: number) => s + v, 0) / 10 },
  { nombre: 'Regresión logística', detalle: 'cambia la semilla de los pliegues', color: '#24398A', v: desvios(S.pliegues), media: S.pliegues.reduce((s: number, v: number) => s + v, 0) / 10 },
]
// diferencia entre el primer y el segundo lugar de la búsqueda (en milésimas)
const C = (datos as any).corridas
const brecha = 1000 * (C[0].auc - C[1].auc)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="yEje" :x-ticks="[-2, -1, 0, 1, 2]" :y-ticks="[]"
          :x-formato="(v: number) => (v > 0 ? '+' : '') + String(v).replace('-', '−')"
          x-titulo="diferencia con el promedio de las diez corridas (milésimas de AUC)" :rejilla-x="true" />
    <line :x1="x(0)" :x2="x(0)" :y1="m.t - 10" :y2="H - m.b" stroke="#5E6678" stroke-width="1.5" />
    <g v-if="referencia">
      <rect :x="x(-brecha / 2)" :y="m.t - 14" :width="x(brecha / 2) - x(-brecha / 2)" :height="H - m.b - m.t + 14" fill="#EFDDAA" opacity="0.55" />
      <text :x="x(-brecha / 2) - 10" :y="m.t + 2" text-anchor="end" style="fill: #A87C12; font-weight: 600; font-size: 17px">ancho: 1.º contra 2.º modelo ({{ brecha.toFixed(1) }} milésimas)</text>
    </g>
    <g v-for="(s, k) in series.slice(0, filas)" :key="k">
      <text :x="m.l - 24" :y="y(k) - 4" text-anchor="end" style="fill: #1A2133; font-weight: 600; font-size: 21px">{{ s.nombre }}</text>
      <text :x="m.l - 24" :y="y(k) + 22" text-anchor="end" style="fill: #5E6678; font-size: 18px">{{ s.detalle }} · AUC ≈ {{ s.media.toFixed(3) }}</text>
      <circle v-for="(v, i) in s.v" :key="i" :cx="x(v)" :cy="y(k) + jit[i] * 110" r="10" :fill="s.color" fill-opacity="0.85" stroke="#fff" stroke-width="2" />
    </g>
  </svg>
</template>
