<script setup lang="ts">
// Escena del experimento de los polinomios. Se usa en varias láminas seguidas,
// cada una agregando una capa: datos → ajuste → datos nuevos → errores.
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u01.json'

const props = withDefaults(defineProps<{
  grado?: number | null, real?: boolean, nuevos?: boolean, residuos?: 'tr' | 'nu' | null,
  ancho?: number, alto?: number, etiqueta?: string,
}>(), { grado: null, real: false, nuevos: false, residuos: null, ancho: 1000, alto: 440 })

const P = datos.polinomios as any
const m = { l: 50, r: 16, t: 14, b: 36 }
const x = computed(() => scaleLinear().domain([0, 1]).range([m.l, props.ancho - m.r]))
const y = computed(() => scaleLinear().domain([-1.9, 1.9]).range([props.alto - m.b, m.t]))
const camino = (ys: number[]) => line<number>().x((_, i) => x.value(P.x[i])).y(v => y.value(v))(ys)!
const id = `c${Math.random().toString(36).slice(2)}`
const segs = computed(() => {
  if (props.grado == null || !props.residuos) return []
  const xs = props.residuos === 'tr' ? P.xtr : P.xnu
  const ys = props.residuos === 'tr' ? P.ytr : P.ynu
  const pr = props.residuos === 'tr' ? P.pred_tr[props.grado] : P.pred_nu[props.grado]
  return xs.map((xv: number, i: number) => ({ x: x.value(xv), y1: y.value(ys[i]), y2: y.value(Math.max(-2.2, Math.min(2.2, pr[i]))) }))
})
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${alto}`" role="img" aria-label="Datos y polinomio ajustado">
    <defs><clipPath :id="id"><rect :x="m.l" :y="m.t" :width="ancho - m.l - m.r" :height="alto - m.t - m.b" /></clipPath></defs>
    <Ejes :x="x" :y="y" :x-ticks="[0, 0.25, 0.5, 0.75, 1]" :y-ticks="[-1, 0, 1]" />
    <g :clip-path="`url(#${id})`">
      <path v-if="real" :d="camino(P.f)" fill="none" stroke="#8A91A0" stroke-width="2.5" stroke-dasharray="2 7" stroke-linecap="round" />
      <line v-for="(s, i) in segs" :key="`r${i}`" :x1="s.x" :x2="s.x" :y1="s.y1" :y2="s.y2" stroke="#D5433C" stroke-width="3" opacity="0.85" />
      <path v-if="grado != null" :d="camino(P.curvas[grado])" fill="none" stroke="#24398A" stroke-width="4" stroke-linejoin="round" />
    </g>
    <circle v-for="(xv, i) in P.xtr" :key="`t${i}`" :cx="x(xv)" :cy="y(P.ytr[i])" r="9" fill="#E1B73D" stroke="#fff" stroke-width="3" />
    <template v-if="nuevos">
      <circle v-for="(xv, i) in P.xnu" :key="`n${i}`" :cx="x(xv)" :cy="y(P.ynu[i])" r="8" fill="#fff" stroke="#20305A" stroke-width="3" />
    </template>
  </svg>
</template>
