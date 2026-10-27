<script setup lang="ts">
// Barras horizontales desde cero, con valor al final y categorías resaltadas.
const props = withDefaults(defineProps<{ datos: Record<string, number>, resaltar?: string[], unidad?: string, max?: number, ancho?: number, notas?: Record<string, string>, izquierda?: number }>(),
  { resaltar: () => [], unidad: '', ancho: 600, notas: () => ({}), izquierda: 170 })
const filas = Object.entries(props.datos)
const alto = 34, x0 = props.izquierda, W = props.ancho
const max = props.max ?? Math.max(...filas.map(f => f[1]))
const x = (v: number) => x0 + (v / max) * (W - x0 - 180)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${filas.length * alto + 10}`" role="img">
    <g v-for="([n, v], i) in filas" :key="n" :transform="`translate(0, ${i * alto})`">
      <text :x="x0 - 12" y="23" text-anchor="end" :style="{ fill: resaltar.includes(n) ? '#B23028' : '#1A2133', fontWeight: resaltar.includes(n) ? 650 : 400 }">{{ n }}</text>
      <rect :x="x0" y="7" :width="x(v) - x0" height="22" rx="3" :fill="resaltar.includes(n) ? '#D5433C' : '#24398A'" />
      <text :x="x(v) + 8" y="24" style="font-size: 16px">{{ v.toLocaleString('es-MX') }}{{ unidad }}<tspan v-if="notas[n]" dx="10" style="fill:#B23028; font-weight: 600">{{ notas[n] }}</tspan></text>
    </g>
  </svg>
</template>
