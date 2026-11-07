<script setup lang="ts">
// Módulo Inception (3a de GoogLeNet): cuatro ramas en paralelo sobre la misma entrada de 192 canales,
// cuyas salidas se apilan. Las convoluciones 1×1 (doradas) reducen canales antes de las caras.
const W = 920, H = 400
const ramas = [
  { y: 50, pasos: [{ k: 1, c: 64, red: false }], sal: 64 },
  { y: 140, pasos: [{ k: 1, c: 96, red: true }, { k: 3, c: 128, red: false }], sal: 128 },
  { y: 230, pasos: [{ k: 1, c: 16, red: true }, { k: 5, c: 32, red: false }], sal: 32 },
  { y: 320, pasos: [{ k: 0, c: 192, red: false }, { k: 1, c: 32, red: true }], sal: 32 },
]
const filtro = (k: number) => (k === 0 ? 3 : k) * 13
const xs = [290, 500]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <!-- entrada -->
    <rect x="30" y="80" width="150" height="240" rx="6" fill="#EFF2F9" stroke="#24398A" stroke-width="1.6" />
    <text x="105" y="196" text-anchor="middle" style="fill: #24398A; font-weight: 700; font-size: 22px">entrada</text>
    <text x="105" y="226" text-anchor="middle" style="font-size: 21px">192 canales</text>
    <!-- salida concatenada -->
    <g>
      <rect x="764" y="40" width="56" :height="64 * 0.62" fill="#24398A" />
      <rect x="764" :y="40 + 64 * 0.62" width="56" :height="128 * 0.62" fill="#496EA5" />
      <rect x="764" :y="40 + 192 * 0.62" width="56" :height="32 * 0.62" fill="#8DA3C9" />
      <rect x="764" :y="40 + 224 * 0.62" width="56" :height="32 * 0.62" fill="#C0CEE3" />
      <text x="832" y="150" style="fill: #1A2133; font-weight: 700; font-size: 22px">se apilan</text>
      <text x="832" y="178" style="font-size: 22px">256</text>
      <text x="832" y="204" style="font-size: 22px">canales</text>
    </g>
    <g v-for="(r, i) in ramas" :key="i">
      <line x1="180" y1="200" :x2="xs[0] - 30" :y2="r.y + 20" stroke="#8A91A0" stroke-width="1.5" />
      <g v-for="(p, j) in r.pasos" :key="j">
        <rect :x="xs[j] - filtro(p.k) / 2" :y="r.y + 20 - filtro(p.k) / 2" :width="filtro(p.k)" :height="filtro(p.k)" rx="3"
              :fill="p.k === 0 ? '#F6F7FA' : p.red ? '#FBF6E6' : '#FFFFFF'"
              :stroke="p.k === 0 ? '#8A91A0' : p.red ? '#C9971C' : '#24398A'" stroke-width="2" />
        <text :x="xs[j] + 40" :y="r.y + 20" style="fill: #1A2133; font-weight: 700; font-size: 24px">{{ p.k === 0 ? 'max-pool' : `${p.k}×${p.k}` }}</text>
        <text :x="xs[j] + 40" :y="r.y + 46" style="font-size: 20px">{{ p.k === 0 ? 'sin parámetros' : `→ ${p.c} canales` }}</text>
        <line v-if="j < r.pasos.length - 1" :x1="xs[j] + filtro(p.k) / 2 + 2" :y1="r.y + 20" :x2="xs[j + 1] - filtro(r.pasos[j + 1].k) / 2 - 2" :y2="r.y + 20" stroke="#8A91A0" stroke-width="1.5" />
      </g>
      <line :x1="(r.pasos.length === 1 ? xs[0] : xs[1]) + 150" :y1="r.y + 20" x2="760" :y2="60 + i * 50" stroke="#8A91A0" stroke-width="1.5" />
    </g>
  </svg>
</template>
