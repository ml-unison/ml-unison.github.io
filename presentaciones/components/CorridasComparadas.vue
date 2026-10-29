<script setup lang="ts">
// Siete corridas registradas en MLflow: AUC (media ± 1 desv. est. entre pliegues) y log-loss, validación cruzada.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u08-01.json'
const C = (datos as any).corridas as { nombre: string, familia: string, auc: number, auc_de: number, logloss: number }[]
withDefaults(defineProps<{ logloss?: boolean }>(), { logloss: true })
const color: Record<string, string> = { logistica: '#24398A', bosque: '#D5433C', boosting: '#C9971C' }
const W = 1120, H = 380, m = { l: 290, t: 50, b: 84 }
const p1 = [m.l, 760], p2 = [830, W - 20]
const xa = scaleLinear().domain([0.70, 0.79]).range(p1)
const xl = scaleLinear().domain([0.40, 0.66]).range(p2)
const fila = (i: number) => m.t + 18 + i * ((H - m.t - m.b - 18) / (C.length - 1))
const yEje = scaleLinear().domain([0, 1]).range([H - m.b + 18, m.t])
const f2 = (v: number) => v.toFixed(2)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <text :x="p1[0]" :y="22" style="fill: #20305A; font-weight: 650; font-size: 20px">AUC (más es mejor)</text>
    <Ejes :x="xa" :y="yEje" :x-ticks="[0.70, 0.72, 0.74, 0.76, 0.78]" :y-ticks="[]" :x-formato="f2" :rejilla-x="true" x-titulo="media ± 1 desv. est. entre 5 pliegues" />
    <g v-for="(c, i) in C" :key="c.nombre">
      <text :x="m.l - 18" :y="fila(i) + 6" text-anchor="end" style="fill: #1A2133; font-size: 19px">{{ c.nombre }}</text>
      <line :x1="xa(c.auc - c.auc_de)" :x2="xa(c.auc + c.auc_de)" :y1="fila(i)" :y2="fila(i)" stroke="#5E6678" stroke-width="2" />
      <line v-for="e in [c.auc - c.auc_de, c.auc + c.auc_de]" :key="e" :x1="xa(e)" :x2="xa(e)" :y1="fila(i) - 7" :y2="fila(i) + 7" stroke="#5E6678" stroke-width="2" />
      <circle :cx="xa(c.auc)" :cy="fila(i)" r="10" :fill="color[c.familia]" stroke="#fff" stroke-width="2" />
      <text :x="xa(c.auc + c.auc_de) + 10" :y="fila(i) + 6" style="fill: #1A2133; font-size: 17px">{{ c.auc.toFixed(3) }}</text>
    </g>
    <g v-if="logloss">
      <text :x="p2[0]" :y="22" style="fill: #20305A; font-weight: 650; font-size: 20px">log-loss (menos es mejor)</text>
      <Ejes :x="xl" :y="yEje" :x-ticks="[0.4, 0.5, 0.6]" :y-ticks="[]" :x-formato="f2" :rejilla-x="true" x-titulo="media en 5 pliegues" />
      <g v-for="(c, i) in C" :key="`l${c.nombre}`">
        <line :x1="p2[0]" :x2="p2[1]" :y1="fila(i)" :y2="fila(i)" stroke="#E2E5EC" stroke-dasharray="3 5" />
        <circle :cx="xl(c.logloss)" :cy="fila(i)" r="10" :fill="color[c.familia]" stroke="#fff" stroke-width="2" />
        <text v-if="c.logloss > 0.55" :x="xl(c.logloss) - 16" :y="fila(i) + 6" text-anchor="end" style="fill: #1A2133; font-size: 17px">{{ c.logloss.toFixed(2) }}</text>
      </g>
    </g>
  </svg>
</template>
