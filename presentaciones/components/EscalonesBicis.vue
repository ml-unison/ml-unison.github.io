<script setup lang="ts">
// Lo que aprendió el árbol podado de las bicicletas: predicción por hora para los dos tipos de día
// (2012, despejado, 25 °C) contra el promedio observado de 2012.
import { scaleLinear } from 'd3-scale'
import { line, curveStepAfter, curveMonotoneX } from 'd3-shape'
import datos from '../datos/u05-01.json'
const props = withDefaults(defineProps<{ observado?: boolean, ancho?: number, alto?: number }>(), { observado: true, ancho: 720, alto: 440 })
const E = (datos as any).escalones
const W = props.ancho, H = props.alto, m = { l: 64, r: 20, t: 34, b: 62 }
const x = scaleLinear().domain([-0.5, 23.5]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 820]).range([H - m.b, m.t])
// escalones centrados en cada hora: se repite el último valor para cerrar el último escalón
const escalon = (v: number[]) => line<number>().curve(curveStepAfter).x((_, i) => x(i - 0.5)).y(t => y(t))([...v, v[v.length - 1]])!
const suave = (v: number[]) => line<number>().curve(curveMonotoneX).x((_, i) => x(i)).y(t => y(t))(v)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[0, 3, 6, 9, 12, 15, 18, 21]" :y-ticks="[0, 200, 400, 600, 800]" x-titulo="hora del día" y-titulo="bicicletas por hora" />
    <template v-if="observado">
      <path :d="suave(E.obs_laboral)" fill="none" stroke="#24398A" stroke-width="1.8" stroke-dasharray="3 4" />
      <path :d="suave(E.obs_descanso)" fill="none" stroke="#D5433C" stroke-width="1.8" stroke-dasharray="3 4" />
    </template>
    <path :d="escalon(E.descanso)" fill="none" stroke="#D5433C" stroke-width="3.2" />
    <path :d="escalon(E.laboral)" fill="none" stroke="#24398A" stroke-width="3.2" />
    <text :x="x(19.8)" :y="y(640)" style="fill: #24398A; font-weight: 700">día laboral</text>
    <text :x="x(10.7)" :y="y(635)" style="fill: #D5433C; font-weight: 700">fin de semana</text>
    <text v-if="observado" :x="x(0)" :y="y(760)" style="font-size: 16px">escalones: árbol podado · punteado: promedio observado (2012)</text>
  </svg>
</template>
