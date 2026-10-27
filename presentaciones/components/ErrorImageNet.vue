<script setup lang="ts">
// Error top-5 en el conjunto de prueba de ImageNet (ILSVRC), según los artículos originales:
// Krizhevsky et al. 2012 (AlexNet y el segundo lugar), Szegedy et al. 2015 (GoogLeNet), He et al. 2016 (ResNet).
import { scaleLinear } from 'd3-scale'
const filas = [
  { anio: '2012', nombre: 'segundo lugar (sin redes)', e: 26.2, color: '#8A91A0' },
  { anio: '2012', nombre: 'AlexNet · 8 capas', e: 15.3, color: '#24398A' },
  { anio: '2014', nombre: 'GoogLeNet · 22 capas', e: 6.67, color: '#24398A' },
  { anio: '2015', nombre: 'ResNet · 152 capas', e: 3.57, color: '#24398A' },
]
const W = 560, H = 300, m = { l: 270, r: 70, t: 8, b: 52 }
const x = scaleLinear().domain([0, 30]).range([m.l, W - m.r])
const banda = (H - m.t - m.b) / filas.length
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <line v-for="t in [0, 10, 20, 30]" :key="t" :x1="x(t)" :x2="x(t)" :y1="m.t" :y2="H - m.b" stroke="#E2E5EC" />
    <text v-for="t in [0, 10, 20, 30]" :key="`t${t}`" :x="x(t)" :y="H - m.b + 24" text-anchor="middle">{{ t }}</text>
    <text :x="(m.l + W - m.r) / 2" :y="H - 4" text-anchor="middle" class="titulo-eje">error top-5 (%)</text>
    <g v-for="(f, k) in filas" :key="k">
      <rect :x="x(0)" :y="m.t + k * banda + 8" :width="x(f.e) - x(0)" :height="banda - 16" :fill="f.color" rx="3" />
      <text :x="m.l - 12" :y="m.t + (k + 0.5) * banda + 6" text-anchor="end" style="fill: #1A2133; font-weight: 600">{{ f.anio }} · {{ f.nombre }}</text>
      <text :x="x(f.e) + 8" :y="m.t + (k + 0.5) * banda + 6" style="fill: #1A2133">{{ String(f.e) }}</text>
    </g>
  </svg>
</template>
