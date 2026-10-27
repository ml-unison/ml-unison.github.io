<script setup lang="ts">
// Softmax sobre 64 claves con q, k ~ N(0, I): peso máximo o tamaño de la jacobiana (botones),
// con y sin dividir los puntajes entre √d_k.
import { ref, computed } from 'vue'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-04.json'
const E = (datos as any).escala
const W = 600, H = 360, m = { l: 64, r: 110, t: 30, b: 66 }
const x = scaleLog().base(2).domain([4, 1024]).range([m.l, W - m.r])
const vistas = {
  max: { t: 'peso de la clave más parecida', sin: E.max_sin, con: E.max_con, y: scaleLinear().domain([0, 1]).range([H - m.b, m.t]), ticks: [0, 0.25, 0.5, 0.75, 1] },
  jac: { t: 'tamaño de la jacobiana del softmax', sin: E.jac_sin, con: E.jac_con, y: scaleLinear().domain([0, 0.32]).range([H - m.b, m.t]), ticks: [0, 0.1, 0.2, 0.3] },
}
const vista = ref<'max' | 'jac'>('max')
const p = computed(() => vistas[vista.value])
const l = (v: number[]) => line<number>().x((_, i) => x(E.dk[i])).y(q => p.value.y(q))(v)!
const fmt = (v: number) => `${v}`
</script>

<template>
  <div>
    <div class="er__botones">
      <button :class="{ activo: vista === 'max' }" @click="vista = 'max'">peso máximo</button>
      <button :class="{ activo: vista === 'jac' }" @click="vista = 'jac'">jacobiana</button>
      <span>{{ p.t }}</span>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <Ejes :key="vista" :x="x" :y="p.y" :x-ticks="E.dk" :y-ticks="p.ticks" :x-formato="fmt" x-titulo="dₖ (dimensión de consultas y claves)" />
      <path :d="l(p.sin)" fill="none" stroke="#D5433C" stroke-width="3" />
      <path :d="l(p.con)" fill="none" stroke="#24398A" stroke-width="3" />
      <circle v-for="(v, i) in p.sin" :key="`s${i}`" :cx="x(E.dk[i])" :cy="p.y(v)" r="5" fill="#D5433C" />
      <circle v-for="(v, i) in p.con" :key="`c${i}`" :cx="x(E.dk[i])" :cy="p.y(v)" r="5" fill="#24398A" />
      <text :x="x(1024) + 12" :y="p.y(p.sin[4]) + 6" class="etiqueta" style="fill: #1A2133">qᵀk</text>
      <text :x="x(1024) + 12" :y="p.y(p.con[4]) + 6" class="etiqueta" style="fill: #1A2133">qᵀk / √dₖ</text>
    </svg>
  </div>
</template>

<style scoped>
.er__botones { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; font-size: 19px; }
.er__botones span { color: var(--marino); font-weight: 600; margin-left: 8px; }
.er__botones button { font: 600 17px var(--f-texto); padding: 3px 13px; border-radius: 16px; border: 2px solid var(--azul); background: #fff; color: var(--azul); cursor: pointer; }
.er__botones button.activo { background: var(--azul); color: #fff; }
</style>
