<script setup lang="ts">
// Exactitud en el Titanic (libreta u01-e01): una fuga de información se ve "demasiado buena".
const filas = [
  { nombre: 'Predecir siempre «no sobrevive»', v: 0.618, tipo: 'base' },
  { nombre: 'Regresión logística (validación cruzada)', v: 0.801, tipo: 'ok' },
  { nombre: 'La misma, agregando la columna «bote»', v: 0.972, tipo: 'fuga' },
]
const W = 1060, x0 = 430, x1 = 980
const x = (v: number) => x0 + v * (x1 - x0)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} 250`" role="img" aria-label="Exactitud de tres modelos en el Titanic">
    <line v-for="t in [0, 0.25, 0.5, 0.75, 1]" :key="t" :x1="x(t)" :x2="x(t)" y1="0" y2="210" stroke="#E2E5EC" />
    <text v-for="t in [0, 0.25, 0.5, 0.75, 1]" :key="`t${t}`" :x="x(t)" y="240" text-anchor="middle" class="eje">{{ t }}</text>
    <g v-for="(f, i) in filas" :key="i" :transform="`translate(0, ${i * 70 + 10})`">
      <text x="0" y="32" class="nombre">{{ f.nombre }}</text>
      <rect :x="x0" y="10" :width="x(f.v) - x0" height="34" rx="4" :fill="f.tipo === 'fuga' ? '#D5433C' : f.tipo === 'base' ? '#C5CAD5' : '#24398A'" />
      <text :x="x(f.v) + 10" y="34" class="valor">{{ f.v.toFixed(2) }}</text>
    </g>
  </svg>
</template>

<style scoped>
.nombre { font: 400 21px var(--f-texto); fill: #1A2133; }
.valor { font: 650 20px var(--f-titulo); fill: #1A2133; }
.eje { font: 400 16px var(--f-texto); fill: #8A91A0; }
</style>
