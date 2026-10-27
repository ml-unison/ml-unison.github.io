<script setup lang="ts">
// Seis clientes con cinco facturas cada uno: partición aleatoria contra partición por cliente.
import { mulberry32 } from './azar'
const colores = ['#24398A', '#D5433C', '#C9971C', '#6B7385', '#496EA5', '#946721']
const r = mulberry32(5)
const facturas = colores.flatMap((_, c) => Array.from({ length: 5 }, () => c))
const aleatorio = facturas.map(() => Math.floor(r() * 3))
const grupos = facturas.map(c => c % 3)
const filas = [['Partición aleatoria (KFold)', aleatorio], ['Partición por cliente (GroupKFold)', grupos]] as const
const W = 760
</script>

<template>
  <div class="pg">
    <div v-for="[t, asig] in filas" :key="t">
      <div class="pg__t">{{ t }}</div>
      <svg class="grafica" :viewBox="`0 0 ${W} 130`">
        <g v-for="p in 3" :key="p" :transform="`translate(${(p - 1) * 255}, 0)`">
          <rect x="0" y="0" width="240" height="120" rx="8" :fill="p === 1 ? '#FBF6E6' : '#F6F7FA'" :stroke="p === 1 ? '#C9971C' : '#E2E5EC'" stroke-width="1.5" />
          <text x="10" y="22" style="font-size: 15px">{{ p === 1 ? 'parte que valida' : `parte ${p}` }}</text>
          <circle v-for="(c, i) in facturas.map((c, i) => ({ c, i })).filter(o => asig[o.i] === p - 1)" :key="i" :cx="22 + (i % 9) * 25" :cy="50 + Math.floor(i / 9) * 26" r="9" :fill="colores[c.c]" />
        </g>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.pg { display: flex; flex-direction: column; gap: 14px; }
.pg__t { font-weight: 600; font-size: 19px; color: var(--marino); }
</style>
