<script setup lang="ts">
// Validación con origen móvil (ventana creciente): a la izquierda, qué meses entrenan y cuál valida en cada pliegue;
// a la derecha, el MAE de validación a 24 h contra el número de árboles de LightGBM.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-03.json'
const C = (datos as any).cv
// línea de tiempo en meses desde enero de 2011 (0) hasta diciembre de 2012 (23)
const Wt = 470, Ht = 300, mt = { l: 20, r: 20, t: 30, b: 50 }
const xt = scaleLinear().domain([0, 24]).range([mt.l, Wt - mt.r])
const filas = [
  { n: 'pliegue 1', v: 18, c: '#24398A' }, { n: 'pliegue 2', v: 19, c: '#D5433C' }, { n: 'pliegue 3', v: 20, c: '#C9971C' },
]
const yf = (k: number) => mt.t + 30 + k * 56
// curvas
const W = 600, H = 380, m = { l: 60, r: 130, t: 30, b: 60 }
const x = scaleLinear().domain([0, 800]).range([m.l, W - m.r])
const y = scaleLinear().domain([35, 72]).range([H - m.b, m.t])
const camino = (v: number[]) => line<number>().x((_, i) => x(C.arboles[i])).y(t => y(t))(v)!
const series = [
  { k: 'julio', c: '#24398A', d: false }, { k: 'agosto', c: '#D5433C', d: false },
  { k: 'septiembre', c: '#C9971C', d: false }, { k: 'promedio', c: '#1A2133', d: true },
]
const etiquetas = (() => {
  const u = series.map(s => ({ ...s, y: y(C[s.k][C[s.k].length - 1]) })).sort((a, b) => a.y - b.y)
  for (let i = 1; i < u.length; i++) u[i].y = Math.max(u[i].y, u[i - 1].y + 24)
  return u
})()
</script>

<template>
  <div class="om">
    <svg class="grafica" :viewBox="`0 0 ${Wt} ${Ht}`">
      <g v-for="(f, k) in filas" :key="k">
        <text :x="mt.l" :y="yf(k) - 8" style="font-size: 16px">{{ f.n }}</text>
        <rect :x="xt(0)" :y="yf(k)" :width="xt(f.v) - xt(0)" height="20" fill="#C0CEE3" />
        <rect :x="xt(f.v)" :y="yf(k)" :width="xt(f.v + 1) - xt(f.v)" height="20" :fill="f.c" />
      </g>
      <text :x="mt.l" :y="yf(3) - 8" style="font-size: 16px">al final</text>
      <rect :x="xt(0)" :y="yf(3)" :width="xt(21) - xt(0)" height="20" fill="#C0CEE3" />
      <rect :x="xt(21)" :y="yf(3)" :width="xt(24) - xt(21)" height="20" fill="#1A2133" />
      <line :x1="xt(0)" :x2="xt(24)" :y1="Ht - mt.b" :y2="Ht - mt.b" stroke="#8A91A0" />
      <text v-for="(t, i) in ['ene 2011', 'jul', 'ene 2012', 'jul', 'ene 2013']" :key="i" :x="xt(6 * i)" :y="Ht - mt.b + 24" text-anchor="middle" style="font-size: 16px">{{ t }}</text>
      <text :x="xt(0)" :y="Ht - 4" style="font-size: 16px; fill: #5E6678">azul claro: entrena · color: valida · negro: prueba</text>
    </svg>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <Ejes :x="x" :y="y" :x-ticks="[0, 200, 400, 600, 800]" :y-ticks="[40, 50, 60, 70]" x-titulo="número de árboles" y-titulo="MAE de validación a 24 h" />
      <line :x1="x(C.elegido)" :x2="x(C.elegido)" :y1="m.t" :y2="H - m.b" stroke="#8A91A0" stroke-dasharray="5 4" />
      <text :x="x(C.elegido) + 6" :y="m.t + 14" style="fill: #1A2133; font-size: 17px">{{ C.elegido }}</text>
      <path v-for="s in series" :key="s.k" :d="camino(C[s.k])" fill="none" :stroke="s.c" stroke-width="3" :stroke-dasharray="s.d ? '8 6' : undefined" />
      <text v-for="e in etiquetas" :key="`e${e.k}`" :x="W - m.r + 10" :y="e.y + 6" style="fill: #1A2133; font-weight: 600">{{ e.k }}</text>
    </svg>
  </div>
</template>

<style scoped>
.om { display: grid; grid-template-columns: 470px 600px; gap: 40px; align-items: center; }
</style>
