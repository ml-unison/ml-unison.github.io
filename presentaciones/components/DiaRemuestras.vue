<script setup lang="ts">
// Martes 29 de marzo de 2011 (prueba): árboles profundos de 8 remuestras contra bagging de esas mismas remuestras.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u05-02.json'
const props = withDefaults(defineProps<{ bagging?: boolean }>(), { bagging: true })
const D = (datos as any).dia
const W = 560, H = 380, m = { l: 62, r: 14, t: 30, b: 58 }
const x = scaleLinear().domain([0, 23]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 520]).range([H - m.b, m.t]).clamp(true)
const camino = (s: number[]) => line<number>().x((_, i) => x(D.hora[i])).y(v => y(v))(s)!
const paneles = [
  { clave: 'arboles', titulo: 'Un árbol profundo por remuestra', color: '#D5433C' },
  { clave: 'bagging', titulo: 'Bagging de 50 árboles, mismas remuestras', color: '#24398A' },
]
</script>

<template>
  <div class="dr">
    <div v-for="(p, k) in paneles" :key="p.clave" :style="{ visibility: k === 1 && !props.bagging ? 'hidden' : 'visible' }">
      <div class="dr__t" :style="{ color: p.color }">{{ p.titulo }}</div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <Ejes :x="x" :y="y" :x-ticks="[0, 4, 8, 12, 16, 20]" :y-ticks="[0, 100, 200, 300, 400, 500]" x-titulo="hora" />
        <path v-for="(s, i) in D[p.clave]" :key="i" :d="camino(s)" fill="none" :stroke="p.color" stroke-width="2" stroke-opacity="0.55" stroke-linejoin="round" />
        <circle v-for="(v, i) in D.real" :key="`r${i}`" :cx="x(D.hora[i])" :cy="y(v)" r="5" fill="#1A2133" stroke="#fff" stroke-width="1.2" />
      </svg>
    </div>
  </div>
</template>

<style scoped>
.dr { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
.dr__t { font-weight: 650; font-size: 21px; margin-bottom: 2px; }
</style>
