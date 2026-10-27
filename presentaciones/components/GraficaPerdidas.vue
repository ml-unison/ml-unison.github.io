<script setup lang="ts">
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import { AZUL, DORADO, GRIS, ROJO } from './colores'

const W = 620, H = 440, m = { l: 56, r: 24, t: 44, b: 74 }
const x = scaleLinear().domain([-2.5, 2.5]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 4]).range([H - m.b, m.t])
const ms = Array.from({ length: 301 }, (_, i) => -2.5 + (5 * i) / 300)
const camino = (f: (v: number) => number) => line<number>().x(v => x(v)).y(v => y(Math.min(f(v), 4.6)))(ms)!

const series = [
  { nombre: '0–1', color: GRIS, d: `M${x(-2.5)},${y(1)}H${x(0)}V${y(0)}H${x(2.5)}`, ancho: 4, lx: 0.45, ly: 2.6 },
  { nombre: 'logística', color: AZUL, d: camino(v => Math.log2(1 + Math.exp(-v))), ancho: 3, lx: 0.45, ly: 3.65 },
  { nombre: 'bisagra', color: DORADO, d: camino(v => Math.max(0, 1 - v)), ancho: 3, lx: 0.45, ly: 3.3},
  { nombre: 'cuadrática', color: ROJO, d: camino(v => (1 - v) ** 2), ancho: 3, punteada: true, lx: 0.45, ly: 2.95 },
]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="Funciones de pérdida en función del margen">
    <defs><clipPath id="area-perdidas"><rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" /></clipPath></defs>
    <Ejes :x="x" :y="y" :x-ticks="[-2, -1, 0, 1, 2]" :y-ticks="[0, 1, 2, 3, 4]" x-titulo="margen  y·s" y-titulo="pérdida" />
    <line :x1="x(0)" :x2="x(0)" :y1="y(0)" :y2="y(4)" stroke="#C5CAD5" stroke-dasharray="4 5" />
    <g clip-path="url(#area-perdidas)">
      <path v-for="s in series" :key="s.nombre" :d="s.d" fill="none" :stroke="s.color" :stroke-width="s.ancho"
            stroke-linejoin="round" stroke-linecap="round" :stroke-dasharray="s.punteada ? '9 7' : undefined" />
    </g>
    <g v-for="s in series" :key="`e${s.nombre}`" :transform="`translate(${x(s.lx)},${y(s.ly)})`">
            <line x1="0" x2="22" y1="-6" y2="-6" :stroke="s.color" :stroke-width="s.ancho" :stroke-dasharray="s.punteada ? '6 4' : undefined" />
      <text class="etiqueta" x="30" y="0" style="fill:#1A2133">{{ s.nombre }}</text>
    </g>
    <text :x="x(0) - 10" :y="y(0) - 12" text-anchor="end" style="font-size:15px">← clasificación incorrecta</text>
    <text :x="x(0) + 10" :y="y(0) - 12" style="font-size:15px">correcta →</text>
  </svg>
</template>
