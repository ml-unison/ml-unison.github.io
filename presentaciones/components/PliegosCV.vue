<script setup lang="ts">
// Validación cruzada con 5 particiones: ¿qué datos vio el escalador en cada iteración?
const props = defineProps<{ correcto?: boolean }>()
const W = 520, x0 = 30, ancho = 90, sep = 6, alto = 26, filas = 5
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} 250`" role="img">
    <g v-if="!correcto">
      <rect :x="x0" y="2" :width="5 * ancho + 4 * sep" height="14" rx="3" fill="#D5433C" opacity="0.85" />
      <text :x="x0" y="34" class="cap">el escalador se ajusta una vez, con todo</text>
    </g>
    <g v-for="i in filas" :key="i" :transform="`translate(0, ${46 + (i - 1) * (alto + 10)})`">
      <text x="0" :y="alto - 7" class="cap">{{ i }}</text>
      <g v-for="j in 5" :key="j">
        <rect :x="x0 + (j - 1) * (ancho + sep)" y="0" :width="ancho" :height="alto" rx="4"
              :fill="i === j ? '#E1B73D' : '#24398A'" :opacity="i === j ? 1 : 0.9" />
        <rect v-if="correcto && i !== j" :x="x0 + (j - 1) * (ancho + sep)" y="-6" :width="ancho" height="4" rx="2" fill="#20305A" />
      </g>
    </g>
  </svg>
</template>

<style scoped>
.cap { font: 400 16px var(--f-texto); fill: #5E6678; }
</style>
