<script setup lang="ts">
// Tiempo del gradiente de una pérdida logística con P parámetros (M = 500 ejemplos):
// modo directo (torch.func.jacfwd) contra modo inverso (jacrev). Datos: herramientas/laminas/u06_01.py.
import { scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-01.json'
const C = (datos as any).costo
const W = 640, H = 420, m = { l: 70, r: 150, t: 40, b: 64 }
const x = scaleLog().domain([8, 4000]).range([m.l, W - m.r])
const y = scaleLog().domain([0.05, 30]).range([H - m.b, m.t])
const camino = (v: number[]) => line<number>().x((_, i) => x(C.P[i])).y(t => y(t))(v)!
const series = [{ k: 'directo', n: 'modo directo', c: '#D5433C' }, { k: 'inverso', n: 'modo inverso', c: '#24398A' }]
const fmt = (v: number) => String(v)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[10, 100, 1000]" :y-ticks="[0.1, 1, 10]" :x-formato="v => v.toLocaleString('es-MX')" :y-formato="fmt"
          x-titulo="número de parámetros P" y-titulo="tiempo del gradiente (ms)" />
    <g v-for="s in series" :key="s.k">
      <path :d="camino(C[s.k])" fill="none" :stroke="s.c" stroke-width="3" />
      <circle v-for="(v, i) in C[s.k]" :key="i" :cx="x(C.P[i])" :cy="y(v)" r="4.5" :fill="s.c" />
      <text :x="x(C.P[C.P.length - 1]) + 14" :y="y(C[s.k][C.P.length - 1]) + 6" style="fill: #1A2133; font-weight: 600">{{ s.n }}</text>
    </g>
  </svg>
</template>
