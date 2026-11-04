<script setup lang="ts">
// Distribución de z = Σ w_j a_j para una neurona con n_in = 256 entradas de varianza 1 (10,000 ejemplos),
// con tres escalas de los pesos. Los valores fuera de [−4, 4] se acumulan en las orillas.
// Datos: herramientas/laminas/u06_01.py.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u06-01.json'
const E = (datos as any).escala
const filas = [
  { k: 'pequena', n: 'w ~ N(0, 0.01²)', c: '#6B7385' },
  { k: 'justa', n: 'w ~ N(0, 1/n_in)', c: '#24398A' },
  { k: 'grande', n: 'w ~ N(0, 1)', c: '#D5433C' },
]
const W = 620, h = 118, gap = 26, m = { l: 18, r: 18 }
const H = filas.length * (h + gap) + 30
const x = scaleLinear().domain([-4, 4]).range([m.l, W - m.r])
const alto = (k: string) => Math.max(...E[k].frec)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <g v-for="(f, j) in filas" :key="f.k" :transform="`translate(0, ${j * (h + gap)})`">
      <rect v-for="(v, i) in E[f.k].frec" :key="i" :x="x(E.bordes[i]) + 0.5" :width="x(E.bordes[i + 1]) - x(E.bordes[i]) - 1"
            :y="22 + h - 6 - (h - 28) * v / alto(f.k)" :height="(h - 28) * v / alto(f.k)" :fill="f.c" opacity="0.85" />
      <line :x1="m.l" :x2="W - m.r" :y1="22 + h - 6" :y2="22 + h - 6" stroke="#8A91A0" stroke-width="1.2" />
      <text :x="m.l" :y="16" style="fill: #1A2133; font-weight: 650">{{ f.n }}</text>
      <text :x="W - m.r" :y="16" text-anchor="end" :style="{ fill: f.c, fontWeight: 700 }">desviación de z: {{ E[f.k].desv }}</text>
    </g>
    <g :transform="`translate(0, ${filas.length * (h + gap) - gap + 46})`">
      <text v-for="t in [-4, -2, 0, 2, 4]" :key="t" :x="x(t)" :y="0" text-anchor="middle">{{ t === -4 ? '≤ −4' : t === 4 ? '≥ 4' : String(t).replace('-', '−') }}</text>
    </g>
  </svg>
</template>
