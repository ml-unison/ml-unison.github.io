<script setup lang="ts">
// Tablero de PSI: atributos (filas) por mes de 2012 (columnas), contra el mismo mes de 2011.
import { scaleLinear } from 'd3-scale'
import d from '../datos/u08-02.json'
const T = (d as any).tablero as { fila: string, psi: number[] }[]
const props = withDefaults(defineProps<{ filas?: number }>(), { filas: 5 })
const meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const W = 1100, izq = 200, celda = (W - izq) / 12, alto = 52
const color = scaleLinear<string>().domain([0, 0.1, 0.25, 0.5]).range(['#FFFFFF', '#C0CEE3', '#496EA5', '#20305A']).clamp(true)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${T.length * alto + 44}`" role="img">
    <text v-for="(mm, j) in meses" :key="mm" :x="izq + (j + 0.5) * celda" y="22" text-anchor="middle">{{ mm }}</text>
    <g v-for="(f, i) in T" :key="f.fila" :transform="`translate(0, ${36 + i * alto})`" :style="{ opacity: i < filas ? 1 : 0, transition: 'opacity .4s' }">
      <line v-if="f.fila === 'rentas observadas'" :x1="0" :x2="W" y1="-1" y2="-1" stroke="#1A2133" stroke-width="1.5" />
      <text :x="izq - 14" :y="alto / 2 + 7" text-anchor="end" :style="{ fill: '#1A2133', fontWeight: f.fila === 'predicción' || f.fila === 'rentas observadas' ? 650 : 400 }">{{ f.fila }}</text>
      <g v-for="(v, j) in f.psi" :key="j">
        <rect :x="izq + j * celda + 1.5" y="1.5" :width="celda - 3" :height="alto - 3" rx="3" :fill="color(v)" />
        <text :x="izq + (j + 0.5) * celda" :y="alto / 2 + 7" text-anchor="middle"
              :style="{ fill: v > 0.25 ? '#FFFFFF' : '#1A2133', fontWeight: v > 0.25 ? 700 : 400, fontSize: '18px' }">{{ v.toFixed(2) }}</text>
      </g>
    </g>
  </svg>
</template>
