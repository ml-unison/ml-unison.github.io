<script setup lang="ts">
// Exactitud por validación cruzada (5 pliegos) de la SVM gaussiana en los dígitos, para cada C y γ.
import datos from '../datos/u04.json'
const M = (datos as any).digitos_svm
// secuencial de un solo tono (azul institucional), de claro a oscuro
const mezcla = (t: number) => { const a = [239, 242, 249], b = [32, 48, 90]; return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})` }
const color = (v: number) => mezcla(Math.max(0, Math.min(1, (v - 0.2) / 0.79)))
const tinta = (v: number) => (v > 0.7 ? '#fff' : '#1A2133')
const c = 92, f = 58
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${110 + c * M.gamma.length} ${60 + f * M.C.length + 40}`">
    <text v-for="(g, j) in M.gamma" :key="`g${j}`" :x="110 + j * c + c / 2" y="40" text-anchor="middle" style="fill: #1A2133">{{ g }}</text>
    <text :x="110 + (c * M.gamma.length) / 2" y="16" text-anchor="middle" class="titulo-eje">γ</text>
    <g v-for="(C, i) in M.C" :key="`c${i}`">
      <text x="96" :y="60 + i * f + f / 2 + 6" text-anchor="end" style="fill: #1A2133">C = {{ C }}</text>
      <g v-for="(g, j) in M.gamma" :key="`${i}-${j}`">
        <rect :x="110 + j * c + 1" :y="60 + i * f + 1" :width="c - 2" :height="f - 2" :fill="color(M.exactitud[i][j])" rx="3"
              :stroke="M.mejor.C === C && M.mejor.gamma === g ? '#C9971C' : 'none'" stroke-width="4" />
        <text :x="110 + j * c + c / 2" :y="60 + i * f + f / 2 + 6" text-anchor="middle" :style="{ fill: tinta(M.exactitud[i][j]), fontWeight: 600, fontSize: '17px' }">{{ (100 * M.exactitud[i][j]).toFixed(1) }}</text>
      </g>
    </g>
  </svg>
</template>
