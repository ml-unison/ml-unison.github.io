<script setup lang="ts">
// Un origen de prueba: las 24 horas previas, lo que ocurrió después y los pronósticos de tres estrategias.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-03.json'
const props = withDefaults(defineProps<{ indice?: number, mostrar?: number }>(), { indice: 4, mostrar: 3 })
const T = (datos as any).trayectorias[props.indice]
const W = 760, H = 400, m = { l: 60, r: 200, t: 30, b: 60 }
const x = scaleLinear().domain([-23, 24]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 700]).range([H - m.b, m.t])
const antes = line<number>().x((_, i) => x(i - 23)).y(v => y(v))(T.antes)!
const despues = (v: number[]) => line<number>().x((_, i) => x(i)).y(t => y(t))([T.antes[23], ...v])!
const series = [
  { k: 'directa', n: 'directa', c: '#24398A', e: 'LightGBM directa' },
  { k: 'recursiva', n: 'recursiva', c: '#D5433C', e: 'LightGBM recursiva' },
  { k: 'red', n: 'red multisalida', c: '#C9971C', e: 'red multisalida' },
]
const f = new Date(T.origen.replace(' ', 'T') + ':00')
const DIA = ['domingo', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado']
const MES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const titulo = `origen: ${DIA[f.getDay()]} ${f.getDate()} de ${MES[f.getMonth()]}, ${f.getHours()} h`
const fmt = (h: number) => { const v = (f.getHours() + h + 48) % 24; return `${v} h` }
const etiquetas = (() => {
  const u = series.map(s => ({ ...s, y: y(T[s.k][23]) })).concat([{ k: 'real', n: 'observado', c: '#1A2133', e: '', y: y(T.real[23]) }]).sort((a, b) => a.y - b.y)
  for (let i = 1; i < u.length; i++) u[i].y = Math.max(u[i].y, u[i - 1].y + 23)
  return u
})()
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[-18, -12, -6, 0, 6, 12, 18, 24]" :x-formato="fmt" :y-ticks="[0, 200, 400, 600]" y-titulo="rentas por hora" />
    <line :x1="x(0)" :x2="x(0)" :y1="m.t" :y2="H - m.b" stroke="#1A2133" stroke-width="1.5" />
    <text :x="x(0) + 6" :y="m.t + 14" style="fill: #1A2133; font-size: 17px; font-weight: 600">{{ titulo }}</text>
    <path :d="antes" fill="none" stroke="#1A2133" stroke-width="2.4" />
    <path :d="despues(T.real)" fill="none" stroke="#1A2133" stroke-width="2.4" stroke-dasharray="3 4" />
    <path v-for="s in series.slice(0, mostrar)" :key="s.k" :d="despues(T[s.k])" fill="none" :stroke="s.c" stroke-width="3" />
    <g v-for="e in etiquetas.filter(e => e.k === 'real' || series.slice(0, mostrar).some(s => s.k === e.k))" :key="`e${e.k}`">
      <text :x="W - m.r + 12" :y="e.y + 6" style="fill: #1A2133; font-weight: 600; font-size: 18px">
        <tspan :style="`fill: ${e.c}`">— </tspan>{{ e.n }}<tspan v-if="e.e" style="font-weight: 400; fill: #5E6678"> · {{ T.err[e.e] }}</tspan>
      </text>
    </g>
  </svg>
</template>
