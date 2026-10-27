<script setup lang="ts">
// Rentas de Capital Bikeshare: dos años por día (tendencia y estación) o dos semanas por hora (ciclos diario y semanal).
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-03.json'
const props = withDefaults(defineProps<{ vista?: 'anios' | 'semanas' }>(), { vista: 'anios' })
const S = (datos as any).serie
const W = 1120, H = 360, m = { l: 70, r: 20, t: 40, b: 50 }
const MES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const DIA = ['dom', 'lun', 'mar', 'mié', 'jue', 'vie', 'sáb']

// vista de dos años
const n = S.diario.length
const xa = scaleLinear().domain([0, n - 1]).range([m.l, W - m.r])
const ya = scaleLinear().domain([0, 9000]).range([H - m.b, m.t])
const dDiario = line<number>().x((_, i) => xa(i)).y(v => ya(v))(S.diario)!
const dMedia = line<number | null>().defined(v => v != null).x((_, i) => xa(i)).y(v => ya(v as number))(S.media)!
const inicio = new Date(S.inicio + 'T00:00:00')
const marcas = [] as { i: number, t: string }[]
for (let k = 0; k < 24; k += 3) {
  const f = new Date(inicio.getFullYear(), k, 1)
  marcas.push({ i: Math.round((f.getTime() - inicio.getTime()) / 864e5), t: `${MES[f.getMonth()]} ${f.getFullYear()}` })
}

// vista de dos semanas
const Y = S.semanas.y as number[]
const xs = scaleLinear().domain([0, Y.length - 1]).range([m.l, W - m.r])
const ys = scaleLinear().domain([0, 1000]).range([H - m.b, m.t])
const dSem = line<number>().x((_, i) => xs(i)).y(v => ys(v))(Y)!
const i0 = new Date(S.semanas.inicio + 'T00:00:00')
const dias = Array.from({ length: Y.length / 24 }, (_, k) => { const f = new Date(i0.getTime() + k * 864e5); return { k, t: `${DIA[f.getDay()]} ${f.getDate()}`, fin: f.getDay() === 0 || f.getDay() === 6 } })
</script>

<template>
  <svg v-if="props.vista === 'anios'" class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <rect :x="xa(S.prueba)" :y="m.t" :width="xa(n - 1) - xa(S.prueba)" :height="H - m.t - m.b" fill="#C0CEE3" opacity="0.55" />
    <text :x="xa(S.prueba) + 8" :y="m.t + 22" style="fill: #1A2133; font-weight: 600">prueba</text>
    <Ejes :x="xa" :y="ya" :x-ticks="[]" :y-ticks="[0, 2000, 4000, 6000, 8000]" :y-formato="v => v.toLocaleString('es-MX')" y-titulo="rentas por día" />
    <text v-for="mk in marcas" :key="mk.i" :x="xa(mk.i)" :y="H - m.b + 28" text-anchor="middle">{{ mk.t }}</text>
    <path :d="dDiario" fill="none" stroke="#8DA3C9" stroke-width="1.3" />
    <path :d="dMedia" fill="none" stroke="#24398A" stroke-width="3.5" />
    <line :x1="xa(S.sandy)" :x2="xa(S.sandy) - 60" :y1="ya(250)" :y2="ya(1200)" stroke="#D5433C" stroke-width="1.5" />
    <text :x="xa(S.sandy) - 64" :y="ya(1200) - 6" text-anchor="end" style="fill: #D5433C; font-weight: 600">huracán Sandy</text>
    <text :x="xa(150)" :y="ya(6900)" style="fill: #1A2133">media de 28 días</text>
  </svg>
  <svg v-else class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <rect v-for="d in dias.filter(d => d.fin)" :key="`f${d.k}`" :x="xs(d.k * 24)" :y="m.t" :width="xs(24) - xs(0)" :height="H - m.t - m.b" fill="#F6F7FA" />
    <Ejes :x="xs" :y="ys" :x-ticks="[]" :y-ticks="[0, 250, 500, 750, 1000]" y-titulo="rentas por hora" />
    <text v-for="d in dias" :key="d.k" :x="xs(d.k * 24 + 12)" :y="H - m.b + 28" text-anchor="middle" :style="d.fin ? 'fill:#A87C12;font-weight:600' : ''">{{ d.t }}</text>
    <path :d="dSem" fill="none" stroke="#24398A" stroke-width="2.2" />
  </svg>
</template>
