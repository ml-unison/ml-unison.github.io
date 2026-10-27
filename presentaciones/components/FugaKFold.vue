<script setup lang="ts">
// MAE a 7 días estimado con KFold aleatorio, con origen móvil y medido en el futuro real (2021–2024).
import { scaleBand, scaleLinear } from 'd3-scale'
import datos from '../datos/u07-01.json'
const props = withDefaults(defineProps<{ grupos?: number }>(), { grupos: 3 })
const F = (datos as any).fuga
const W = 760, H = 400, m = { l: 62, r: 130, t: 50, b: 56 }
const cols = [{ k: 'kfold', n: 'KFold aleatorio' }, { k: 'movil', n: 'origen móvil' }, { k: 'futuro', n: 'futuro real 2021–2024' }]
const x = scaleBand<string>().domain(cols.map(c => c.k)).range([m.l, W - m.r]).paddingInner(0.3).paddingOuter(0.15)
const y = scaleLinear().domain([0, 2.6]).range([H - m.b, m.t])
const bw = x.bandwidth() / 2
const colores = ['#24398A', '#D5433C']
</script>

<template>
  <div>
    <div class="fuga__leyenda">
      <span v-for="(f, j) in F.tabla" :key="f.modelo"><i :style="{ background: colores[j] }" />{{ f.modelo }}</span>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
      <Ejes :x="scaleLinear().domain([0, 1]).range([m.l, W - m.r])" :y="y" :x-ticks="[]" :y-ticks="[0, 0.5, 1, 1.5, 2, 2.5]" y-titulo="MAE a 7 días (°C)" />
      <line :x1="m.l" :x2="W - m.r + 4" :y1="y(F.clima_h7)" :y2="y(F.clima_h7)" stroke="#1A2133" stroke-width="1.6" stroke-dasharray="7 6" />
      <g v-for="(c, i) in cols.slice(0, grupos)" :key="c.k">
        <g v-for="(f, j) in F.tabla" :key="f.modelo">
          <rect :x="x(c.k)! + j * bw" :width="bw - 4" :y="y(f[c.k])" :height="y(0) - y(f[c.k])" :fill="colores[j]" />
          <text :x="x(c.k)! + j * bw + (bw - 4) / 2" :y="y(f[c.k]) - 8" paint-order="stroke" stroke="#fff" stroke-width="5" text-anchor="middle" style="fill: #1A2133; font-weight: 600">{{ f[c.k].toFixed(2) }}</text>
        </g>
        <text :x="x(c.k)! + bw" :y="H - m.b + 30" text-anchor="middle" style="fill: #1A2133; font-weight: 600">{{ c.n }}</text>
      </g>
      <text :x="W - m.r + 10" :y="y(F.clima_h7) - 4" style="fill: #1A2133; font-weight: 600">climatología</text>
      <text :x="W - m.r + 10" :y="y(F.clima_h7) + 20" style="fill: #1A2133">{{ F.clima_h7.toFixed(2) }} °C</text>
    </svg>
  </div>
</template>

<style scoped>
.fuga__leyenda { display: flex; gap: 26px; font-size: 20px; color: var(--tinta); margin-bottom: 4px; }
.fuga__leyenda i { display: inline-block; width: 18px; height: 18px; border-radius: 3px; margin-right: 8px; vertical-align: -2px; }
</style>
