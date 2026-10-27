<script setup lang="ts">
// Importancias en el censo con dos columnas inútiles (id, ruido): MDI (entrenamiento) contra permutación (prueba).
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u05-02.json'
const props = withDefaults(defineProps<{ permutacion?: boolean }>(), { permutacion: true })
const I = (datos as any).importancias
const n = I.nombres.length
const W = 560, fila = 25, m = { l: 150, r: 30, t: 6, b: 62 }
const H = m.t + fila * n + m.b
const xm = scaleLinear().domain([0, 0.13]).range([m.l, W - m.r])
const xp = scaleLinear().domain([-0.002, 0.05]).range([m.l, W - m.r])
const inutil = (s: string) => s === 'id' || s === 'ruido'
const paneles = [
  { t: 'Impureza media (MDI), entrenamiento', x: xm, v: I.mdi, ticks: [0, 0.04, 0.08, 0.12], tit: 'importancia (suman 1)' },
  { t: 'Permutación, prueba', x: xp, v: I.perm, ticks: [0, 0.01, 0.02, 0.03, 0.04, 0.05], tit: 'caída de la AUC' },
]
</script>

<template>
  <div class="id">
    <div v-for="(p, k) in paneles" :key="k" :style="{ visibility: k === 1 && !props.permutacion ? 'hidden' : 'visible' }">
      <div class="id__t">{{ p.t }}</div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <g class="rejilla"><line v-for="t in p.ticks" :key="t" :x1="p.x(t)" :x2="p.x(t)" :y1="m.t" :y2="H - m.b" /></g>
        <g class="eje"><text v-for="t in p.ticks" :key="t" :x="p.x(t)" :y="H - m.b + 24" text-anchor="middle">{{ t }}</text></g>
        <text class="titulo-eje" :x="(m.l + W - m.r) / 2" :y="H - m.b + 50" text-anchor="middle">{{ p.tit }}</text>
        <g v-for="(nom, i) in I.nombres" :key="nom">
          <text :x="m.l - 10" :y="m.t + i * fila + 18" text-anchor="end" :style="{ fill: inutil(nom) ? '#B23028' : '#1A2133', fontWeight: inutil(nom) ? 700 : 400, fontSize: '17px' }">{{ nom }}</text>
          <rect :x="Math.min(p.x(0), p.x(p.v[i]))" :y="m.t + i * fila + 5" :width="Math.abs(p.x(p.v[i]) - p.x(0))" :height="fila - 9" :fill="inutil(nom) ? '#D5433C' : '#24398A'" />
        </g>
        <line :x1="p.x(0)" :x2="p.x(0)" :y1="m.t" :y2="H - m.b" stroke="#8A91A0" stroke-width="1.2" />
      </svg>
    </div>
  </div>
</template>

<style scoped>
.id { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
.id__t { font-weight: 650; color: var(--marino); font-size: 20px; margin-bottom: 4px; }
</style>
