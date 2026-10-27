<script setup lang="ts">
// Pronóstico del CO₂ desde diciembre de 2014: SARIMA con intervalos del 80 % y 95 % contra lo observado; opcionalmente, otros modelos.
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line, area } from 'd3-shape'
import datos from '../datos/u07-02.json'
const props = withDefaults(defineProps<{ otros?: boolean, observado?: boolean, ancho?: number, alto?: number }>(), { otros: false, observado: true, ancho: 760, alto: 440 })
const P = (datos as any).pronostico
const i0 = P.inicio
const tp: number[] = P.t.slice(i0)
const W = props.ancho, H = props.alto, m = { l: 62, r: 175, t: 30, b: 50 }
const x = scaleLinear().domain([2010, 2026.8]).range([m.l, W - m.r])
const y = scaleLinear().domain([385, 435]).range([H - m.b, m.t])
const banda = (lo: number[], hi: number[]) => area<number>().x((_, i) => x(tp[i])).y0((_, i) => y(lo[i])).y1(v => y(v))(hi)!
const lin = (t: number[], v: number[]) => line<number>().x((_, i) => x(t[i])).y(u => y(u))(v)!
const otrosM = [
  { k: 'cuadratica', n: 'cuadrática + mes', c: '#C9971C' },
  { k: 'lineal', n: 'lineal + mes', c: '#D5433C' },
  { k: 'ingenuo', n: 'estacional ingenuo', c: '#8A91A0' },
]
const fin = (v: number[]) => y(v[v.length - 1])
const etiquetas = computed(() => {
  const e = [{ n: 'SARIMA', c: '#24398A', y: fin(P.media) }]
  if (props.observado) e.push({ n: 'observado', c: '#1A2133', y: fin(P.obs) })
  if (props.otros) for (const o of otrosM) e.push({ n: o.n, c: o.c, y: fin(P.otros[o.k]) })
  e.sort((a, b) => a.y - b.y)
  for (let i = 1; i < e.length; i++) e[i].y = Math.max(e[i].y, e[i - 1].y + 26)
  return e
})
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <Ejes :x="x" :y="y" :x-ticks="[2010, 2014, 2018, 2022, 2026]" :y-ticks="[390, 400, 410, 420, 430]" :x-formato="v => String(v)" y-titulo="CO₂ (ppm)" />
    <path :d="banda(P.lo95, P.hi95)" fill="#C0CEE3" opacity="0.55" />
    <path :d="banda(P.lo80, P.hi80)" fill="#C0CEE3" />
    <line :x1="x(2015)" :x2="x(2015)" :y1="m.t" :y2="H - m.b" stroke="#5E6678" stroke-dasharray="5 5" />
    <path :d="lin(P.t.slice(0, i0 + 1), P.obs.slice(0, i0 + 1))" fill="none" stroke="#1A2133" stroke-width="2" />
    <template v-if="otros">
      <path v-for="o in otrosM" :key="o.k" :d="lin(tp, P.otros[o.k])" fill="none" :stroke="o.c" stroke-width="2" />
    </template>
    <path :d="lin(tp, P.media)" fill="none" stroke="#24398A" stroke-width="2.6" />
    <path v-if="observado" :d="lin(tp, P.obs.slice(i0))" fill="none" stroke="#1A2133" stroke-width="2" />
    <g v-for="e in etiquetas" :key="e.n">
      <line :x1="W - m.r + 10" :x2="W - m.r + 26" :y1="e.y" :y2="e.y" :stroke="e.c" stroke-width="3.2" />
      <text :x="W - m.r + 32" :y="e.y + 6" style="fill: #1A2133; font-weight: 600">{{ e.n }}</text>
    </g>
  </svg>
</template>
