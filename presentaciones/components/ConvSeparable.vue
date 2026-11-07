<script setup lang="ts">
// Convolución normal contra separable en profundidad, con C = 3 canales de entrada y C' = 4 de salida.
// Arriba: cuatro filtros de 3×3×3 (cada uno mezcla espacio y canales a la vez).
// Abajo: un filtro de 3×3 por canal (solo espacio) y después cuatro filtros de 1×1×3 (solo canales).
const CAN = ['#24398A', '#C9971C', '#D5433C']
const W = 860, H = 420
const s = 16       // lado de una celda
// un filtro k×k con `capas` canales dibujado como cubo (capas desplazadas)
const cubo = (x: number, y: number, k: number, capas: string[]) =>
  capas.map((col, j) => ({ x: x + j * 7, y: y - j * 7, lado: k * s, col }))
const normal = [0, 1, 2, 3].map(i => cubo(180 + i * 100, 70, 3, CAN))
const prof = [0, 1, 2].map(i => cubo(180 + i * 75, 280, 3, [CAN[i]]))
const punto = [0, 1, 2, 3].map(i => cubo(500 + i * 52, 300, 1, CAN))
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <text x="0" y="100" style="fill: #1A2133; font-weight: 700; font-size: 26px">normal</text>
    
    <g v-for="(f, i) in normal" :key="`n${i}`">
      <rect v-for="(c, j) in f" :key="j" :x="c.x" :y="c.y" :width="c.lado" :height="c.lado" fill="#FFFFFF" :stroke="c.col" stroke-width="2" />
    </g>
    <text x="600" y="100" style="fill: #1A2133; font-weight: 700; font-size: 24px">4 · 3 · 9 = 108</text>

    <line x1="0" y1="200" x2="860" y2="200" stroke="#E2E5EC" />

    <text x="0" y="310" style="fill: #1A2133; font-weight: 700; font-size: 26px">separable</text>
    
    <g v-for="(f, i) in prof" :key="`p${i}`">
      <rect v-for="(c, j) in f" :key="j" :x="c.x" :y="c.y" :width="c.lado" :height="c.lado" fill="#FFFFFF" :stroke="c.col" stroke-width="2" />
    </g>
    <text x="255" y="385" text-anchor="middle" style="font-size: 22px">por canal: 3 · 9 = 27</text>
    <text x="440" y="314" text-anchor="middle" style="fill: #1A2133; font-weight: 800; font-size: 26px">→</text>
    <g v-for="(f, i) in punto" :key="`q${i}`">
      <rect v-for="(c, j) in f" :key="j" :x="c.x" :y="c.y" :width="c.lado + 4" :height="c.lado + 4" fill="#FFFFFF" :stroke="c.col" stroke-width="2" />
    </g>
    <text x="590" y="385" text-anchor="middle" style="font-size: 22px">1×1: 4 · 3 = 12</text>
    <text x="740" y="314" style="fill: #1A2133; font-weight: 700; font-size: 24px">= 39</text>
  </svg>
</template>
