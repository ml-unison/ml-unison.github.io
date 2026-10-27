<script setup lang="ts">
// Dependencia parcial (azul) y 60 curvas ICE (gris) del modelo de boosting, en dólares (escala log).
import { ref, computed } from 'vue'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u05-04.json'
const D = datos as any
const props = withDefaults(defineProps<{ inicial?: 'area' | 'anio', botones?: boolean }>(), { inicial: 'area', botones: true })
const atr = ref(props.inicial)
const verICE = ref(true)
const W = 680, H = 430, m = { l: 66, r: 56, t: 34, b: 64 }
const y = scaleLog().domain([45, 560]).range([H - m.b, m.t]).clamp(true)
const P = computed(() => D[`pdp_${atr.value}`])
const x = computed(() => scaleLinear().domain([P.value.malla[0], P.value.malla[P.value.malla.length - 1]]).range([m.l, W - m.r]))
const camino = (s: number[]) => line<number>().x((_, i) => x.value(P.value.malla[i])).y(v => y(v))(s)!
const xTicks = computed(() => atr.value === 'area' ? [100, 150, 200, 250] : [1900, 1925, 1950, 1975, 2000])
</script>

<template>
  <div>
    <div v-if="botones" class="ice__botones">
      <button :class="{ activo: atr === 'area' }" @click="atr = 'area'">área habitable</button>
      <button :class="{ activo: atr === 'anio' }" @click="atr = 'anio'">año de construcción</button>
      <button :class="{ activo: verICE }" @click="verICE = !verICE">curvas ICE</button>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <Ejes :x="x" :y="y" :x-ticks="xTicks" :y-ticks="[50, 75, 100, 150, 200, 300, 400]" :x-formato="(v: number) => String(v)"
            :x-titulo="atr === 'area' ? 'área habitable (m²)' : 'año de construcción'" y-titulo="precio predicho (miles de dólares, escala log)" />
      <g v-if="verICE"><path v-for="(c, i) in P.ice" :key="`${atr}${i}`" :d="camino(c)" fill="none" stroke="#6B7385" stroke-width="1.2" opacity="0.45" /></g>
      <path :d="camino(P.pdp)" fill="none" stroke="#24398A" stroke-width="5" />
      <text :x="W - m.r + 6" :y="y(P.pdp[P.pdp.length - 1]) + 6" style="fill: #24398A; font-weight: 650">PDP</text>
    </svg>
  </div>
</template>

<style scoped>
.ice__botones { display: flex; gap: 10px; margin-bottom: 4px; }
.ice__botones button { font: 600 17px var(--f-texto); padding: 4px 14px; border-radius: 16px; border: 1.5px solid var(--azul); color: var(--azul); background: #fff; cursor: pointer; }
.ice__botones button.activo { background: var(--azul); color: #fff; }
</style>
