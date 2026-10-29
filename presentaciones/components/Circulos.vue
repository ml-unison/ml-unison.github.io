<script setup lang="ts">
// Dos anillos: no se separan con una recta en (x₁, x₂), pero sí con un umbral en z = x₁² + x₂².
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u04.json'
const C = (datos as any).circulos
const S = 380
const a = scaleLinear().domain([-1.35, 1.35]).range([16, S - 16]), b = scaleLinear().domain([-1.35, 1.35]).range([S - 16, 16])
const W2 = 460, H2 = 380
const x2 = scaleLinear().domain([-1.35, 1.35]).range([60, W2 - 16]), z = scaleLinear().domain([0, 1.8]).range([H2 - 56, 20])
</script>

<template>
  <div class="ci">
    <svg class="grafica" :viewBox="`0 0 ${S} ${S}`">
      <rect x="16" y="16" :width="S - 32" :height="S - 32" fill="#F6F7FA" rx="8" />
      <circle v-for="(v, i) in C.x" :key="i" :cx="a(v)" :cy="b(C.y[i])" r="5" :fill="C.g[i] ? '#D5433C' : '#24398A'" opacity="0.85" />
    </svg>
    <div class="ci__f">→<div class="ci__t">Φ(x) = (x₁, x₂, x₁² + x₂²)</div></div>
    <svg class="grafica" :viewBox="`0 0 ${W2} ${H2}`">
      <Ejes :x="x2" :y="z" :x-ticks="[-1, 0, 1]" :y-ticks="[0, 0.5, 1, 1.5]" x-titulo="x₁" y-titulo="z = x₁² + x₂²" />
      <line :x1="x2(-1.35)" :x2="x2(1.35)" :y1="z(0.5)" :y2="z(0.5)" stroke="#C9971C" stroke-width="3" />
      <circle v-for="(v, i) in C.x" :key="i" :cx="x2(v)" :cy="z(C.r2[i])" r="5" :fill="C.g[i] ? '#D5433C' : '#24398A'" opacity="0.85" />
    </svg>
  </div>
</template>

<style scoped>
.ci { display: grid; grid-template-columns: 380px 150px 460px; gap: 20px; align-items: center; justify-content: center; }
.ci__f { font-size: 48px; color: var(--azul); text-align: center; }
.ci__t { font-size: 17px; color: var(--tinta); }
</style>
