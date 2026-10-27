<script setup lang="ts">
// Correlograma (ACF o PACF) con bandas ±1.96/√T; opcionalmente la ACF teórica (puntos) y los rezagos estacionales en rojo.
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'

const props = withDefaults(defineProps<{
  valores: number[], banda: number, teorica?: number[] | null, periodo?: number, titulo?: string,
  ancho?: number, alto?: number, yDominio?: [number, number], xTicks?: number[], yTicks?: number[],
}>(), { ancho: 520, alto: 240, yDominio: () => [-1, 1] as [number, number], periodo: 0 })

const m = { l: 58, r: 14, t: 34, b: 40 }
const K = computed(() => props.valores.length)
const x = computed(() => scaleLinear().domain([0, K.value + 0.6]).range([m.l, props.ancho - m.r]))
const y = computed(() => scaleLinear().domain(props.yDominio).range([props.alto - m.b, m.t]).clamp(true))
const tx = computed(() => props.xTicks ?? (K.value > 30 ? [0, 12, 24, 36, 48].filter(v => v <= K.value) : [1, 5, 10, 15, 20].filter(v => v <= K.value)))
const ty = computed(() => props.yTicks ?? [props.yDominio[0], 0, props.yDominio[1]])
const grosor = computed(() => Math.max(2.5, Math.min(7, 0.45 * (props.ancho - m.l - m.r) / K.value)))
const f = (v: number) => String(v).replace('-', '−')
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${alto}`" role="img">
    <text v-if="titulo" :x="m.l" :y="18" style="fill: #1A2133; font-weight: 600; font-size: 19px">{{ titulo }}</text>
    <rect :x="m.l" :width="ancho - m.l - m.r" :y="y(banda)" :height="y(-banda) - y(banda)" fill="#C0CEE3" opacity="0.6" />
    <g class="rejilla"><line v-for="t in ty" :key="t" :x1="m.l" :x2="ancho - m.r" :y1="y(t)" :y2="y(t)" /></g>
    <line :x1="m.l" :x2="ancho - m.r" :y1="y(0)" :y2="y(0)" stroke="#8A91A0" stroke-width="1.2" />
    <line v-for="(v, i) in valores" :key="i" :x1="x(i + 1)" :x2="x(i + 1)" :y1="y(0)" :y2="y(v)"
          :stroke="periodo && (i + 1) % periodo === 0 ? '#D5433C' : '#24398A'" :stroke-width="grosor" />
    <template v-if="teorica"><circle v-for="(v, i) in teorica" :key="`t${i}`" :cx="x(i + 1)" :cy="y(v)" r="5" fill="#C9971C" stroke="#fff" stroke-width="1.5" /></template>
    <g class="eje">
      <text v-for="t in tx" :key="`x${t}`" :x="x(t)" :y="alto - m.b + 24" text-anchor="middle">{{ t }}</text>
      <text v-for="t in ty" :key="`y${t}`" :x="m.l - 10" :y="y(t) + 6" text-anchor="end">{{ f(t) }}</text>
    </g>
  </svg>
</template>
