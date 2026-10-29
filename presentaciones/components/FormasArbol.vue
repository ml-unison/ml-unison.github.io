<script setup lang="ts">
// Tres formas de hacer crecer un árbol con tres divisiones (cuatro hojas).
// niveles: XGBoost; hojas: LightGBM (siempre la hoja de mayor ganancia); simetrico: CatBoost (la misma pregunta en todo un nivel).
import { AZUL, DORADO, MARINO } from './colores'
const props = withDefaults(defineProps<{ tipo?: 'niveles' | 'hojas' | 'simetrico', ancho?: number }>(), { tipo: 'niveles', ancho: 380 })
type Nodo = { x: number, y: number, t: string }
const formas: Record<string, { nodos: Nodo[], hojas: Nodo[], aristas: number[][], pie: string }> = {
  niveles: {
    nodos: [{ x: 190, y: 34, t: '1' }, { x: 100, y: 112, t: '2' }, { x: 280, y: 112, t: '3' }],
    hojas: [{ x: 55, y: 190, t: '' }, { x: 145, y: 190, t: '' }, { x: 235, y: 190, t: '' }, { x: 325, y: 190, t: '' }],
    aristas: [[190, 34, 100, 112], [190, 34, 280, 112], [100, 112, 55, 190], [100, 112, 145, 190], [280, 112, 235, 190], [280, 112, 325, 190]],
    pie: 'por niveles: todo un nivel antes del siguiente',
  },
  hojas: {
    nodos: [{ x: 120, y: 34, t: '1' }, { x: 210, y: 96, t: '2' }, { x: 280, y: 158, t: '3' }],
    hojas: [{ x: 50, y: 96, t: '' }, { x: 140, y: 158, t: '' }, { x: 230, y: 214, t: '' }, { x: 330, y: 214, t: '' }],
    aristas: [[120, 34, 50, 96], [120, 34, 210, 96], [210, 96, 140, 158], [210, 96, 280, 158], [280, 158, 230, 214], [280, 158, 330, 214]],
    pie: 'por hojas: siempre la de mayor ganancia',
  },
  simetrico: {
    nodos: [{ x: 190, y: 34, t: 'x₁ > 2' }, { x: 100, y: 112, t: 'x₃ > 5' }, { x: 280, y: 112, t: 'x₃ > 5' }],
    hojas: [{ x: 55, y: 190, t: '00' }, { x: 145, y: 190, t: '01' }, { x: 235, y: 190, t: '10' }, { x: 325, y: 190, t: '11' }],
    aristas: [[190, 34, 100, 112], [190, 34, 280, 112], [100, 112, 55, 190], [100, 112, 145, 190], [280, 112, 235, 190], [280, 112, 325, 190]],
    pie: 'simétrico: la misma pregunta en todo el nivel',
  },
}
const F = formas[props.tipo]
const simetrico = props.tipo === 'simetrico'
</script>

<template>
  <div class="fa" :style="{ width: `${ancho}px` }">
    <svg class="grafica" viewBox="0 0 380 236">
      <line v-for="(a, k) in F.aristas" :key="k" :x1="a[0]" :y1="a[1]" :x2="a[2]" :y2="a[3]" :stroke="MARINO" stroke-width="2" />
      <template v-for="(n, k) in F.nodos" :key="`n${k}`">
        <rect v-if="simetrico" :x="n.x - 46" :y="n.y - 17" width="92" height="34" rx="8" :fill="AZUL" />
        <circle v-else :cx="n.x" :cy="n.y" r="19" :fill="AZUL" />
        <text :x="n.x" :y="n.y + 7" text-anchor="middle" fill="#fff" style="font-size: 20px; font-weight: 700">{{ n.t }}</text>
      </template>
      <template v-for="(h, k) in F.hojas" :key="`h${k}`">
        <rect :x="h.x - 22" :y="h.y - 15" width="44" height="30" rx="5" :fill="DORADO" />
        <text v-if="h.t" :x="h.x" :y="h.y + 7" text-anchor="middle" :fill="MARINO" style="font-size: 18px; font-weight: 700">{{ h.t }}</text>
      </template>
    </svg>
    <div class="fa__pie">{{ F.pie }}</div>
  </div>
</template>

<style scoped>
.fa { display: flex; flex-direction: column; align-items: center; }
.fa__pie { font-size: 18px; color: var(--marino); text-align: center; margin-top: 2px; }
</style>
