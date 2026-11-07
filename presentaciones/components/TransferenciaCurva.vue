<script setup lang="ts">
// EuroSAT: exactitud de prueba contra imágenes etiquetadas por clase, para las tres formas de transferir
// una ResNet-18 de ImageNet y una red pequeña entrenada desde cero. Datos: herramientas/laminas/u06_04.py.
import { scaleLog, scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-04.json'
const E = (datos as any).eurosat
const ks: number[] = E.k
const W = 640, H = 400, m = { l: 60, r: 230, t: 16, b: 70 }
const x = scaleLog().domain([ks[0] * 0.8, ks[ks.length - 1] * 1.25]).range([m.l, W - m.r])
const y = scaleLinear().domain([40, 100]).range([H - m.b, m.t])
const series = [
  { nombre: 'ajuste fino', v: E.fino, color: '#24398A', valores: true, dy: -14 },
  { nombre: 'solo la última capa', v: E.ultima, color: '#C9971C', valores: false, dy: 0 },
  { nombre: 'atributos + logística', v: E.transfer, color: '#8A91A0', valores: false, dy: 0 },
  { nombre: 'desde cero', v: E.cero, color: '#D5433C', valores: true, dy: 28 },
]
// etiquetas al final de cada curva, separadas al menos 24 px
const fin = series.map(s => y(100 * s.v[s.v.length - 1]))
const etq: number[] = []
fin.forEach((v, i) => etq.push(i === 0 ? v : Math.max(v, etq[i - 1] + 24)))
const d = (v: number[]) => line<number>().x((_, i) => x(ks[i])).y(a => y(100 * a))(v)!
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="ks" :y-ticks="[40, 60, 80, 100]" x-titulo="imágenes etiquetadas por clase" y-titulo="exactitud de prueba (%)" />
    <g v-for="(s, j) in series" :key="s.nombre">
      <path :d="d(s.v)" fill="none" :stroke="s.color" stroke-width="3.5" />
      <g v-for="(a, i) in s.v" :key="i">
        <circle :cx="x(ks[i])" :cy="y(100 * a)" r="6" :fill="s.color" />
        <text v-if="s.valores" :x="x(ks[i])" :y="y(100 * a) + s.dy" text-anchor="middle" style="fill: #1A2133; font-size: 17px">{{ (100 * a).toFixed(1) }}</text>
      </g>
      <text :x="x(ks[ks.length - 1]) + 22" :y="etq[j] + 6" :style="`fill: ${s.color === '#8A91A0' ? '#5E6678' : s.color === '#C9971C' ? '#A87C12' : s.color}; font-weight: 700`">{{ s.nombre }}</text>
    </g>
  </svg>
</template>
