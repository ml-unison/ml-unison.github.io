<script setup lang="ts">
// Imagen de 8×8. Modo «gris» (0–16, tinta) o «divergente» (rojo negativo, azul positivo, centrado en 0).
const props = withDefaults(defineProps<{ px: number[], tam?: number, modo?: 'gris' | 'divergente' }>(), { tam: 90, modo: 'gris' })
const maxAbs = Math.max(...props.px.map(Math.abs)) || 1
const color = (v: number) => {
  if (props.modo === 'gris') return `rgba(32,48,90,${Math.max(0, Math.min(1, v / 16))})`
  const t = v / maxAbs
  return t >= 0 ? `rgba(36,57,138,${t})` : `rgba(213,67,60,${-t})`
}
</script>

<template>
  <svg :width="tam" :height="tam" viewBox="0 0 8 8" style="display: block; background: #F6F7FA; border-radius: 6px">
    <rect v-for="(v, i) in px" :key="i" :x="i % 8" :y="Math.floor(i / 8)" width="1.02" height="1.02" :fill="color(v)" />
  </svg>
</template>
