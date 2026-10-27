<script setup lang="ts">
// 25 ajustes de un polinomio de grado g (cada uno con otros 15 datos) en gris, su promedio ḡ en azul y f punteada.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u03.json'
const S = (datos as any).sesgo_varianza
const props = withDefaults(defineProps<{ grado: number, ancho?: number, alto?: number }>(), { ancho: 400, alto: 300 })
const M = S.muestras[props.grado]
const m = { l: 10, r: 10, t: 36, b: 10 }
const x = scaleLinear().domain([0, 1]).range([m.l, props.ancho - m.r])
const y = scaleLinear().domain([-2, 2]).range([props.alto - m.b, m.t]).clamp(true)
const c = (v: number[]) => line<number>().x((_, i) => x(S.x[i])).y(d => y(d))(v)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${alto}`">
    <rect :x="m.l" :y="m.t" :width="ancho - m.l - m.r" :height="alto - m.t - m.b" fill="#F6F7FA" rx="6" />
    <text :x="m.l" :y="24" class="titulo-eje" style="font-weight: 600">grado {{ grado }}</text>
    <path v-for="(g, i) in M.curvas" :key="i" :d="c(g)" fill="none" stroke="#8A91A0" stroke-width="1.2" opacity="0.55" />
    <path :d="c(S.f)" fill="none" stroke="#1A2133" stroke-width="2.4" stroke-dasharray="6 5" />
    <path :d="c(M.media)" fill="none" stroke="#24398A" stroke-width="3.4" />
  </svg>
</template>
