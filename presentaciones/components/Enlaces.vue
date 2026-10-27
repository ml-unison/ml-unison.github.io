<script setup lang="ts">
// Cuatro maneras de medir la distancia entre dos grupos.
const A = [[60, 120], [95, 90], [80, 160], [120, 140]]
const B = [[230, 80], [270, 120], [250, 160], [300, 95]]
const cen = (P: number[][]) => [P.reduce((s, p) => s + p[0], 0) / P.length, P.reduce((s, p) => s + p[1], 0) / P.length]
const d = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1])
const pares = A.flatMap(a => B.map(b => [a, b]))
const minimo = pares.reduce((m, p) => d(p[0], p[1]) < d(m[0], m[1]) ? p : m)
const maximo = pares.reduce((m, p) => d(p[0], p[1]) > d(m[0], m[1]) ? p : m)
const casos = [
  { n: 'Simple', t: 'el par más cercano', seg: [minimo] },
  { n: 'Completo', t: 'el par más lejano', seg: [maximo] },
  { n: 'Promedio', t: 'el promedio de todos los pares', seg: pares },
  { n: 'Ward', t: 'cuánto crece la suma de distancias² al centroide al unirlos', seg: [[cen(A), cen(B)]] },
]
</script>

<template>
  <div class="en">
    <div v-for="c in casos" :key="c.n">
      <svg class="grafica" viewBox="0 0 360 230">
        <line v-for="(s, i) in c.seg" :key="i" :x1="s[0][0]" :y1="s[0][1]" :x2="s[1][0]" :y2="s[1][1]" stroke="#C9971C" :stroke-width="c.seg.length > 1 ? 1.2 : 3" :opacity="c.seg.length > 1 ? 0.7 : 1" />
        <circle v-for="(p, i) in A" :key="`a${i}`" :cx="p[0]" :cy="p[1]" r="9" fill="#24398A" />
        <circle v-for="(p, i) in B" :key="`b${i}`" :cx="p[0]" :cy="p[1]" r="9" fill="#D5433C" />
        <template v-if="c.n === 'Ward'">
          <g v-for="(p, i) in [cen(A), cen(B)]" :key="`c${i}`" :transform="`translate(${p[0]},${p[1]})`"><circle r="9" fill="#fff" stroke="#1A2133" stroke-width="2" /><path d="M-4,-4L4,4M-4,4L4,-4" stroke="#1A2133" stroke-width="2" /></g>
        </template>
      </svg>
      <div class="en__n">{{ c.n }}</div>
      <div class="en__t">{{ c.t }}</div>
    </div>
  </div>
</template>

<style scoped>
.en { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; }
.en__n { font: 650 16px var(--f-titulo); letter-spacing: .08em; text-transform: uppercase; color: var(--azul); margin-top: 6px; }
.en__t { font-size: 19px; color: var(--tinta); line-height: 1.3; }
</style>
