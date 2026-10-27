<script setup lang="ts">
// Norma del gradiente de la pérdida de la última letra respecto del encaje de cada letra anterior
// (relativa a la de la propia posición, escala log), promedio de 64 ventanas de validación.
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-04.json'
const G = (datos as any).gradiente
const props = withDefaults(defineProps<{ series?: string[] }>(), { series: () => ['lstm_ini', 'lstm'] })
const info: Record<string, { c: string, t: string, en: number }> = {
  lstm_ini: { c: '#6B7385', t: 'LSTM sin entrenar', en: 38 },
  lstm: { c: '#24398A', t: 'LSTM entrenada', en: 70 },
  tf: { c: '#D5433C', t: 'transformer entrenado', en: 70 },
}
const W = 620, H = 400, m = { l: 70, r: 20, t: 34, b: 70 }
const x = scaleLinear().domain([0, 127]).range([m.l, W - m.r])
const lo = computed(() => Math.floor(Math.min(...props.series.flatMap(k => G[k].filter((v: number | null) => v !== null)))))
const y = computed(() => scaleLinear().domain([Math.max(lo.value, -24), 0.3]).range([H - m.b, m.t]))
const ticks = computed(() => { const p = lo.value < -10 ? 5 : 1; const t = []; for (let v = 0; v >= Math.max(lo.value, -24); v -= p) t.push(v); return t })
const curvas = computed(() => props.series.map(k => ({
  k, ...info[k], v: G[k],
  d: line<number | null>().defined(v => v !== null).x((_, i) => x(i)).y(v => y.value(v as number))(G[k])!,
})))
const pot = (v: number) => v === 0 ? '1' : `10${String(v).replace('-', '⁻').replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d])}`
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :key="lo" :x="x" :y="y" :x-ticks="[0, 20, 40, 60, 80, 100, 120]" :y-ticks="ticks" :y-formato="pot" x-titulo="distancia T − t (letras hacia atrás)" y-titulo="influencia en el gradiente (relativa)" />
    <path v-for="c in curvas" :key="c.k" :d="c.d" fill="none" :stroke="c.c" stroke-width="3" />
    <text v-for="c in curvas" :key="`t${c.k}`" :x="x(c.k === 'lstm' && series.includes('tf') ? 18 : c.en)" :y="c.k === 'lstm' && series.includes('tf') ? y(lo + 0.3) : y(c.v[c.en]) - 22" class="etiqueta" :style="{ fill: c.c }">{{ c.t }}</text>
  </svg>
</template>
