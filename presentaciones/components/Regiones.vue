<script setup lang="ts">
// Regiones de decisión precalculadas en Python, comprimidas en franjas [fila, j0, j1, clase].
const props = withDefaults(defineProps<{ datos: { x: number[], y: number[], franjas: number[][] }, x: any, y: any, colores?: string[], opacidad?: number }>(),
  { colores: () => ['#C0CEE3', '#F3C7C4', '#EFDDAA'], opacidad: 0.75 })
const dx = (props.datos.x[1] - props.datos.x[0]) / 2, dy = (props.datos.y[1] - props.datos.y[0]) / 2
const rects = props.datos.franjas.map(([i, j0, j1, c]) => {
  const x0 = props.x(props.datos.x[j0] - dx), x1 = props.x(props.datos.x[j1] + dx)
  const ya = props.y(props.datos.y[i] + dy), yb = props.y(props.datos.y[i] - dy)
  return { x: x0, w: x1 - x0 + 0.6, y: Math.min(ya, yb), h: Math.abs(yb - ya) + 0.6, c }
})
</script>

<template>
  <g :opacity="opacidad"><rect v-for="(r, k) in rects" :key="k" :x="r.x" :y="r.y" :width="r.w" :height="r.h" :fill="colores[r.c]" /></g>
</template>
