<script setup lang="ts">
// Varianza del promedio de B estimaciones con correlación ρ: fórmula (líneas) y simulación (puntos).
import datos from '../datos/u05-02.json'
const D = (datos as any).promedio
const Bf = Array.from(new Set(Array.from({ length: 160 }, (_, i) => Math.round(Math.pow(10, 3 * i / 159))))).sort((a, b) => a - b)
const series: Record<string, number[]> = {}
for (const rho of D.rhos) series[`ρ = ${rho}`] = Bf.map(B => rho + (1 - rho) / B)
const col = ['#24398A', '#D5433C', '#C9971C', '#6B7385']
</script>

<template>
  <LineasLog :x="Bf" :series="series" x-log :y-dominio="[0.0008, 1.2]" :y-ticks="[0.001, 0.01, 0.1, 1]" :x-dominio="[1, 1000]" :x-ticks="[1, 10, 100, 1000]"
             :formato-y="(v: number) => String(v)" x-titulo="número de estimaciones promediadas, B" y-titulo="Var(promedio) / σ²"
             :ancho="640" :alto="430" :derecha="100">
    <template #default="{ x, y }">
      <g v-for="(rho, k) in D.rhos" :key="rho">
        <line v-if="rho > 0" :x1="x(1)" :x2="x(1000)" :y1="y(rho)" :y2="y(rho)" :stroke="col[k]" stroke-width="1.2" stroke-dasharray="3 4" />
        <circle v-for="(v, i) in D.simulada[k]" :key="i" :cx="x(D.B[i])" :cy="y(v)" r="6" :fill="col[k]" stroke="#fff" stroke-width="1.5" />
      </g>
    </template>
  </LineasLog>
</template>
