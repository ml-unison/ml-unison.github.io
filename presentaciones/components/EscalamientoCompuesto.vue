<script setup lang="ts">
// Las tres formas de agrandar una red (idea de Tan y Le, 2019): más canales (ancho), más capas
// (profundidad), imágenes más grandes (resolución), o las tres a la vez con un mismo coeficiente φ.
// Cada red se dibuja como una pila de capas (barras) sobre su imagen de entrada (cuadro gris).
const W = 1180, H = 380
const base = { capas: 4, ancho: 60, res: 54 }
const redes = [
  { t: 'base', capas: 4, ancho: 60, res: 54, col: '#8A91A0' },
  { t: 'más ancha', capas: 4, ancho: 110, res: 54, col: '#24398A' },
  { t: 'más profunda', capas: 8, ancho: 60, res: 54, col: '#24398A' },
  { t: 'más resolución', capas: 4, ancho: 60, res: 92, col: '#24398A' },
  { t: 'compuesta', capas: 6, ancho: 84, res: 72, col: '#C9971C' },
]
const piso = 330
const alto = 22, sep = 8
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <g v-for="(r, i) in redes" :key="i" :transform="`translate(${120 + i * 235}, 0)`">
      <rect :x="-r.res / 2" :y="piso - r.res" :width="r.res" :height="r.res" fill="#E2E5EC" rx="3" />
      <rect v-for="k in r.capas" :key="k" :x="-r.ancho / 2" :y="piso - r.res - 14 - k * (alto + sep) + sep"
            :width="r.ancho" :height="alto" rx="4" fill="#FFFFFF" :stroke="r.col" stroke-width="2.2" />
      <text x="0" :y="piso + 34" text-anchor="middle" :style="`fill: ${r.col === '#8A91A0' ? '#1A2133' : r.col === '#C9971C' ? '#A87C12' : '#24398A'}; font-weight: 700`">{{ r.t }}</text>
    </g>
  </svg>
</template>
