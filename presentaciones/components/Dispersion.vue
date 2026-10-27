<script setup lang="ts">
// Diagrama de dispersión genérico: puntos coloreados por grupo (−1 = ruido, en gris),
// centros opcionales y etiquetas directas. Sin ejes por omisión (datos estandarizados).
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { SERIE } from './colores'

const props = withDefaults(defineProps<{
  x: number[], y: number[], g?: number[], colores?: readonly string[], ancho?: number, alto?: number,
  radio?: number, opacidad?: number | number[], centros?: number[][], etiquetas?: { x: number, y: number, texto: string, ancla?: string }[],
  ejes?: boolean, xTitulo?: string, yTitulo?: string, titulo?: string, margen?: number, dominio?: number[],
}>(), { ancho: 560, alto: 420, radio: 4.5, opacidad: 0.8, ejes: false, margen: 14 })

const m = computed(() => props.ejes ? { l: 60, r: 16, t: props.titulo ? 46 : 24, b: 64 } : { l: props.margen, r: props.margen, t: props.titulo ? 40 : props.margen, b: props.margen })
const ext = (v: number[]) => { let a = Infinity, b = -Infinity; for (const t of v) { if (t < a) a = t; if (t > b) b = t } const p = (b - a) * 0.04; return [a - p, b + p] }
const x = computed(() => scaleLinear().domain(props.dominio ? [props.dominio[0], props.dominio[1]] : ext(props.x)).range([m.value.l, props.ancho - m.value.r]))
const y = computed(() => scaleLinear().domain(props.dominio ? [props.dominio[2], props.dominio[3]] : ext(props.y)).range([props.alto - m.value.b, m.value.t]))
const col = (i: number) => { const g = props.g?.[i] ?? 0; return g < 0 ? '#B9BEC9' : (props.colores ?? SERIE)[g % (props.colores ?? SERIE).length] }
const op = (i: number) => Array.isArray(props.opacidad) ? props.opacidad[i] : props.opacidad
// el ruido se dibuja primero, para que quede debajo
const orden = computed(() => props.x.map((_, i) => i).sort((a, b) => ((props.g?.[a] ?? 0) < 0 ? -1 : 0) - ((props.g?.[b] ?? 0) < 0 ? -1 : 0)))
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${alto}`" role="img">
    <text v-if="titulo" :x="m.l" :y="22" class="titulo-eje" style="font-weight: 600">{{ titulo }}</text>
    <Ejes v-if="ejes" :x="x" :y="y" :x-ticks="4" :y-ticks="4" :x-titulo="xTitulo" :y-titulo="yTitulo" />
    <circle v-for="i in orden" :key="i" :cx="x(props.x[i])" :cy="y(props.y[i])" :r="radio" :fill="col(i)" :opacity="op(i)" />
    <slot :x="x" :y="y" />
    <g v-for="(c, k) in centros ?? []" :key="`c${k}`" :transform="`translate(${x(c[0])},${y(c[1])})`">
      <circle r="11" fill="#fff" :stroke="(colores ?? SERIE)[k % (colores ?? SERIE).length]" stroke-width="3" />
      <path d="M-5,-5L5,5M-5,5L5,-5" stroke="#1A2133" stroke-width="2.5" />
    </g>
    <text v-for="(e, k) in etiquetas ?? []" :key="`e${k}`" :x="x(e.x)" :y="y(e.y)" :text-anchor="e.ancla ?? 'middle'" class="etiqueta" style="fill: #1A2133">{{ e.texto }}</text>
  </svg>
</template>
