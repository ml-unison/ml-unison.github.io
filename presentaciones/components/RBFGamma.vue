<script setup lang="ts">
// SVM con kernel gaussiano en las lunas, con γ creciente: subajuste, buen ajuste, sobreajuste.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u04.json'
const L = (datos as any).lunas
const W = 380, H = 280
const x = scaleLinear().domain([-1.6, 2.6]).range([0, W]), y = scaleLinear().domain([-1.2, 1.7]).range([H, 0])
const paneles = Object.keys(L.rbf).sort((a, b) => Number(a) - Number(b))
</script>

<template>
  <div class="rg">
    <div v-for="g in paneles" :key="g">
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" style="border-radius: 8px">
        <Regiones :datos="L.rbf[g]" :x="x" :y="y" :colores="['#C0CEE3', '#F3C7C4']" :opacidad="0.8" />
        <circle v-for="(v, i) in L.x" :key="i" :cx="x(v)" :cy="y(L.y[i])" r="4" :fill="L.g[i] ? '#D5433C' : '#24398A'" stroke="#fff" stroke-width="0.8" />
      </svg>
      <div class="rg__t">γ = {{ g }} · {{ L.rbf[g].n_vs }} VS · CV {{ (100 * L.rbf[g].cv).toFixed(1) }} %</div>
    </div>
  </div>
</template>

<style scoped>
.rg { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; }
.rg__t { text-align: center; font-weight: 600; font-size: 19px; color: var(--marino); margin-top: 6px; }
</style>
