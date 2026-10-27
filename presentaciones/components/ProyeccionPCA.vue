<script setup lang="ts">
// Aleta contra masa (estandarizadas): la dirección de la primera componente y,
// con `proyectar`, la proyección de una muestra de puntos sobre ella.
import { computed } from 'vue'
import datos from '../datos/u02.json'
const P = (datos as any).pca2d
const props = withDefaults(defineProps<{ proyectar?: boolean }>(), { proyectar: false })
const u = P.comp[0], v = P.comp[1]
const proy = computed(() => P.muestra.map((i: number) => { const t = P.x[i] * u[0] + P.y[i] * u[1]; return { p: [P.x[i], P.y[i]], q: [t * u[0], t * u[1]] } }))
</script>

<template>
  <Dispersion :x="P.x" :y="P.y" :g="P.x.map(() => 3)" :opacidad="proyectar ? 0.25 : 0.6" :dominio="[-3, 3, -3, 3]" :ancho="600" :alto="470" ejes x-titulo="aleta (estandarizada)" y-titulo="masa (estandarizada)">
    <template #default="{ x, y }">
      <line :x1="x(-3 * u[0])" :y1="y(-3 * u[1])" :x2="x(3 * u[0])" :y2="y(3 * u[1])" stroke="#24398A" stroke-width="3" />
      <line :x1="x(-1.2 * v[0])" :y1="y(-1.2 * v[1])" :x2="x(1.2 * v[0])" :y2="y(1.2 * v[1])" stroke="#24398A" stroke-width="2" stroke-dasharray="6 5" />
      <text :x="x(2.6 * u[0]) + 10" :y="y(2.6 * u[1]) + 30" class="etiqueta" style="fill: #1A2133">PC1 · {{ Math.round(P.var[0] * 100) }} %</text>
      <text :x="x(-1.2 * v[0]) + 10" :y="y(-1.2 * v[1]) + 4" class="etiqueta" style="fill: #1A2133">PC2 · {{ Math.round(P.var[1] * 100) }} %</text>
      <template v-if="proyectar">
        <line v-for="(s, i) in proy" :key="i" :x1="x(s.p[0])" :y1="y(s.p[1])" :x2="x(s.q[0])" :y2="y(s.q[1])" stroke="#D5433C" stroke-width="1.3" />
        <circle v-for="(s, i) in proy" :key="`q${i}`" :cx="x(s.q[0])" :cy="y(s.q[1])" r="4" fill="#D5433C" />
      </template>
    </template>
  </Dispersion>
</template>
