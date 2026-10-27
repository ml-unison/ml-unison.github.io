<script setup lang="ts">
// XGBoost en el mercadeo bancario: entropía cruzada de validación (y, opcionalmente, de entrenamiento)
// contra el número de árboles para tres tasas de aprendizaje; el punto marca el mínimo (parada temprana).
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u05-03.json'
const C = (datos as any).curvas as any[]
const props = withDefaults(defineProps<{ entrena?: boolean }>(), { entrena: true })
const W = 760, H = 430, m = { l: 66, r: 110, t: 30, b: 60 }
const x = scaleLinear().domain([0, 1500]).range([m.l, W - m.r])
const y = scaleLinear().domain([0.2, 0.36]).range([H - m.b, m.t])
const colores = ['#D5433C', '#24398A', '#C9971C']
const camino = (c: any, k: string) => line<number>().x((_, i) => x(c.n[i])).y(v => y(v))(c[k])!
const etiq = C.map((c, k) => ({ k, eta: c.eta, y: Math.max(y(Math.min(c.final, 0.36)), m.t + 6) }))
etiq.sort((a, b) => a.y - b.y)
for (let i = 1; i < etiq.length; i++) etiq[i].y = Math.max(etiq[i].y, etiq[i - 1].y + 26)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <clipPath id="ct-clip"><rect :x="m.l" :y="m.t - 4" :width="W - m.l - m.r" :height="H - m.t - m.b + 4" /></clipPath>
    <Ejes :x="x" :y="y" :x-ticks="[0, 300, 600, 900, 1200, 1500]" :y-ticks="[0.2, 0.24, 0.28, 0.32, 0.36]" x-titulo="número de árboles"
          y-titulo="entropía cruzada" :y-formato="(v: number) => v.toFixed(2)" />
    <g clip-path="url(#ct-clip)">
      <template v-for="(c, k) in C" :key="k">
        <path v-if="entrena" :d="camino(c, 'entrena')" fill="none" :stroke="colores[k]" stroke-width="2" stroke-dasharray="3 5" />
        <path :d="camino(c, 'valida')" fill="none" :stroke="colores[k]" stroke-width="3.4" />
        <circle :cx="x(c.mejor_n)" :cy="y(c.mejor)" r="7" :fill="colores[k]" stroke="#fff" stroke-width="2" />
      </template>
    </g>
    <g v-for="e in etiq" :key="e.k">
      <line :x1="W - m.r + 8" :x2="W - m.r + 24" :y1="e.y" :y2="e.y" :stroke="colores[e.k]" stroke-width="3.4" />
      <text :x="W - m.r + 30" :y="e.y + 6" style="fill: #1A2133; font-weight: 600">η = {{ e.eta }}</text>
    </g>
    <text v-if="entrena" :x="x(1480)" :y="y(0.212)" text-anchor="end" style="font-size: 17px">punteadas: entrenamiento</text>
    <text :x="x(1480)" :y="y(0.348)" text-anchor="end" style="font-size: 17px">continuas: validación · ● mínimo</text>
  </svg>
</template>
