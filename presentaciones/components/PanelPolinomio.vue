<script setup lang="ts">
// Un panel: función real, datos de entrenamiento y polinomio ajustado de un grado dado.
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u01.json'
import { AZUL, DORADO_CLARO, GRIS } from './colores'

const props = withDefaults(defineProps<{ grado: number, ancho?: number, alto?: number, ejeY?: boolean, leyenda?: boolean, conReal?: boolean }>(),
  { ancho: 380, alto: 300, ejeY: true, leyenda: false, conReal: false })
const P = datos.polinomios
const m = computed(() => ({ l: props.ejeY ? 46 : 14, r: 10, t: 10, b: 40 }))
const x = computed(() => scaleLinear().domain([0, 1]).range([m.value.l, props.ancho - m.value.r]))
const y = computed(() => scaleLinear().domain([-2, 2]).range([props.alto - m.value.b, m.value.t]))
const camino = (ys: number[]) => line<number>().x((_, i) => x.value(P.x[i])).y(v => y.value(v))(ys)!
const real = computed(() => camino(P.f))
const ajuste = computed(() => camino(P.curvas[props.grado]))
const id = `clip-${Math.random().toString(36).slice(2)}`
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${alto}`" role="img" :aria-label="`Polinomio de grado ${grado} ajustado a 15 puntos`">
    <defs><clipPath :id="id"><rect :x="m.l" :y="m.t" :width="ancho - m.l - m.r" :height="alto - m.t - m.b" /></clipPath></defs>
    <Ejes :x="x" :y="y" :x-ticks="[0, 0.5, 1]" :y-ticks="ejeY ? [-2, -1, 0, 1, 2] : []" />
    <g :clip-path="`url(#${id})`">
      <path v-if="conReal" :d="real" fill="none" :stroke="GRIS" stroke-width="2.5" stroke-dasharray="3 6" stroke-linecap="round" />
      <path :d="ajuste" fill="none" :stroke="AZUL" stroke-width="3.5" stroke-linejoin="round" style="transition: d .35s ease" />
    </g>
    <circle v-for="(xv, i) in P.xtr" :key="i" :cx="x(xv)" :cy="y(P.ytr[i])" r="7.5" :fill="DORADO_CLARO" stroke="#fff" stroke-width="2.5" />
  </svg>
</template>
