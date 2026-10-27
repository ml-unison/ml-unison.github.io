<script setup lang="ts">
// Valor SHAP del área contra el área (casas de prueba), coloreado por la calidad general; punteada: PDP desplazada.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u05-04.json'
import { divergente } from './casasAmes'
const D = (datos as any).dependencia
const props = withDefaults(defineProps<{ pdp?: boolean }>(), { pdp: true })
const W = 700, H = 440, m = { l: 70, r: 24, t: 36, b: 64 }
const x = scaleLinear().domain([25, 300]).range([m.l, W - m.r]), y = scaleLinear().domain([-0.34, 0.34]).range([H - m.b, m.t])
const curva = line<number>().x((_, i) => x(D.malla[i])).y(v => y(v))(D.pdp)!
const orden = D.area.map((_: number, i: number) => i).sort((a: number, b: number) => D.calidad[a] - D.calidad[b])
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[50, 100, 150, 200, 250, 300]" :y-ticks="[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]" x-titulo="área habitable (m²)" y-titulo="valor SHAP del área" />
    <line :x1="m.l" :x2="W - m.r" :y1="y(0)" :y2="y(0)" stroke="#5E6678" stroke-width="1.2" />
    <circle v-for="i in orden" :key="i" :cx="x(D.area[i])" :cy="y(D.phi[i])" r="5" :fill="divergente((D.calidad[i] - 3) / 6)" stroke="#fff" stroke-width="0.6" />
    <path v-if="pdp" :d="curva" fill="none" stroke="#1A2133" stroke-width="2.6" stroke-dasharray="8 6" />
    <g :transform="`translate(${m.l + 14}, ${m.t + 6})`">
      <text x="0" y="0" style="font-size: 17px">calidad general:</text>
      <template v-for="(q, i) in [3, 5, 7, 9]" :key="q">
        <circle :cx="140 + i * 44" cy="-6" r="6" :fill="divergente((q - 3) / 6)" /><text :x="150 + i * 44" y="0" style="font-size: 17px">{{ q }}</text>
      </template>
      <template v-if="pdp"><line x1="0" x2="30" y1="24" y2="24" stroke="#1A2133" stroke-width="2.6" stroke-dasharray="8 6" /><text x="38" y="30" style="font-size: 17px">dependencia parcial (desplazada)</text></template>
    </g>
  </svg>
</template>
