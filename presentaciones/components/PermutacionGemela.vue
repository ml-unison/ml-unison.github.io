<script setup lang="ts">
// Importancia por permutación en prueba (aumento de la RECM); con `gemela`, al lado el modelo con el área del vendedor.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u05-04.json'
import { NOMBRES } from './casasAmes'
const D = datos as any
const props = withDefaults(defineProps<{ gemela?: boolean }>(), { gemela: false })
const W = 540, fila = props.gemela ? 23 : 33, m = { l: 175, r: 60, t: 10, b: 52 }
const x = scaleLinear().domain([0, 0.08]).range([m.l, W - m.r])
const paneles = props.gemela
  ? [{ t: 'Modelo original', f: D.permutacion as [string, number][] }, { t: 'Con el área declarada por el vendedor', f: D.permutacion_gemela as [string, number][] }]
  : [{ t: '', f: D.permutacion as [string, number][] }]
const alto = (n: number) => m.t + n * fila + m.b
const roja = (k: string, i: number) => props.gemela && i === 1 && (k === 'area' || k === 'area_vendedor')
</script>

<template>
  <div :class="gemela ? 'pg pg--dos' : 'pg'">
    <div v-for="(p, i) in paneles" :key="i">
      <div v-if="p.t" class="pg__t">{{ p.t }}</div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${alto(13)}`">
        <g class="rejilla"><line v-for="t in [0.02, 0.04, 0.06, 0.08]" :key="t" :x1="x(t)" :x2="x(t)" :y1="m.t" :y2="m.t + 13 * fila" /></g>
        <g v-for="([k, v], j) in p.f" :key="k" :transform="`translate(0, ${m.t + j * fila})`">
          <text :x="m.l - 10" y="22" text-anchor="end" :style="{ fill: roja(k, i) ? '#B23028' : '#1A2133', fontWeight: roja(k, i) ? 650 : 400 }">{{ NOMBRES[k] }}</text>
          <rect :x="m.l" y="7" :width="Math.max(1, x(v) - m.l)" height="21" rx="3" :fill="roja(k, i) ? '#D5433C' : '#24398A'" />
          <text :x="x(v) + 7" y="23" style="font-size: 16px; fill: #1A2133">{{ v.toFixed(3) }}</text>
        </g>
        <g class="eje">
          <line :x1="m.l" :x2="W - m.r" :y1="m.t + 13 * fila" :y2="m.t + 13 * fila" />
          <text v-for="t in [0, 0.02, 0.04, 0.06, 0.08]" :key="t" :x="x(t)" :y="m.t + 13 * fila + 24" text-anchor="middle" style="font-size: 16px">{{ t }}</text>
          <text class="titulo-eje" :x="(m.l + W - m.r) / 2" :y="alto(13) - 4" text-anchor="middle" style="font-size: 17px">aumento de la RECM al barajar</text>
        </g>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.pg--dos { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
.pg__t { font-weight: 650; font-size: 20px; color: var(--marino); margin: 0 0 2px 120px; }
</style>
