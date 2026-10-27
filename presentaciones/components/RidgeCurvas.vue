<script setup lang="ts">
// Un mismo conjunto de 15 datos, polinomio de grado 12 con distintos λ.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u03.json'
const R = (datos as any).regularizacion, S = (datos as any).sesgo_varianza
const lams = ['0.0', '0.0001', '0.01', '1.0']
const etiquetas: Record<string, string> = { '0.0': 'λ = 0', '0.0001': 'λ = 10⁻⁴', '0.01': 'λ = 10⁻²', '1.0': 'λ = 1' }
const W = 280, H = 230, m = { l: 6, r: 6, t: 34, b: 6 }
const x = scaleLinear().domain([0, 1]).range([m.l, W - m.r])
const y = scaleLinear().domain([-2, 2]).range([H - m.b, m.t]).clamp(true)
const c = (v: number[]) => line<number>().x((_, i) => x(S.x[i])).y(d => y(d))(v)!
</script>

<template>
  <div class="rc">
    <svg v-for="l in lams" :key="l" class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" fill="#F6F7FA" rx="6" />
      <text :x="m.l" :y="22" class="titulo-eje" style="font-weight: 600">{{ etiquetas[l] }}</text>
      <path :d="c(S.f)" fill="none" stroke="#1A2133" stroke-width="2" stroke-dasharray="6 5" />
      <path :d="c(R.curvas[l])" fill="none" stroke="#24398A" stroke-width="3" />
      <circle v-for="(xx, i) in R.xtr" :key="i" :cx="x(xx)" :cy="y(R.ytr[i])" r="5" fill="#D5433C" stroke="#fff" stroke-width="1.5" />
    </svg>
  </div>
</template>

<style scoped>.rc { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; }</style>
