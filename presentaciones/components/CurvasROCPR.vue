<script setup lang="ts">
// Curvas ROC y precisión–exhaustividad de los dos modelos en los datos de prueba.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u03.json'
const C = (datos as any).credito
const W = 520, H = 440, m = { l: 64, r: 16, t: 20, b: 64 }
const x = scaleLinear().domain([0, 1]).range([m.l, W - m.r]), y = scaleLinear().domain([0, 1]).range([H - m.b, m.t])
const c = (xs: number[], ys: number[]) => line<number>().x((_, i) => x(xs[i])).y(v => y(v))(ys)!
const mods = [['bosque', '#24398A', 'bosque aleatorio'], ['logistica', '#D5433C', 'regresión logística']]
</script>

<template>
  <div class="rp">
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <Ejes :x="x" :y="y" :x-ticks="[0, 0.5, 1]" :y-ticks="[0, 0.5, 1]" x-titulo="tasa de falsos positivos" y-titulo="tasa de verdaderos positivos" />
      <line :x1="x(0)" :y1="y(0)" :x2="x(1)" :y2="y(1)" stroke="#C5C9D3" stroke-width="1.5" stroke-dasharray="5 4" />
      <path v-for="[k, col] in mods" :key="k" :d="c(C.roc[k].fpr, C.roc[k].tpr)" fill="none" :stroke="col" stroke-width="3" />
      <text :x="x(0.22)" :y="y(0.9)" style="fill: #1A2133; font-weight: 600">bosque · AUC {{ C.roc.bosque.auc }}</text>
      <text :x="x(0.5)" :y="y(0.38)" style="fill: #1A2133; font-weight: 600">logística · AUC {{ C.roc.logistica.auc }}</text>
      <text :x="x(0.74)" :y="y(0.62)" style="font-size: 15px">azar · 0.5</text>
    </svg>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <Ejes :x="x" :y="y" :x-ticks="[0, 0.5, 1]" :y-ticks="[0, 0.5, 1]" x-titulo="exhaustividad" y-titulo="precisión" />
      <line :x1="x(0)" :x2="x(1)" :y1="y(C.prevalencia)" :y2="y(C.prevalencia)" stroke="#C5C9D3" stroke-width="1.5" stroke-dasharray="5 4" />
      <text :x="x(0.55)" :y="y(C.prevalencia) - 8" style="font-size: 15px">azar · prevalencia {{ (C.prevalencia * 100).toFixed(0) }} %</text>
      <path v-for="[k, col] in mods" :key="k" :d="c(C.pr[k].exhaustividad, C.pr[k].precision)" fill="none" :stroke="col" stroke-width="3" />
    </svg>
  </div>
</template>

<style scoped>.rp { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }</style>
