<script setup lang="ts">
// Rejilla y ejes para gráficas SVG. Recibe escalas de d3-scale ya construidas.
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  x: any, y: any,
  xTicks?: number | number[], yTicks?: number | number[],
  xFormato?: (v: number) => string, yFormato?: (v: number) => string,
  xTitulo?: string, yTitulo?: string,
  rejillaX?: boolean,
}>(), { xTicks: 5, yTicks: 5, rejillaX: false })

const [x0, x1] = computed(() => props.x.range()).value
const [y0, y1] = computed(() => props.y.range()).value
const tx = computed(() => Array.isArray(props.xTicks) ? props.xTicks : props.x.ticks(props.xTicks))
const ty = computed(() => Array.isArray(props.yTicks) ? props.yTicks : props.y.ticks(props.yTicks))
const fx = (v: number) => props.xFormato ? props.xFormato(v) : String(v).replace('-', '−')
const fy = (v: number) => props.yFormato ? props.yFormato(v) : String(v).replace('-', '−')
</script>

<template>
  <g>
    <g class="rejilla">
      <line v-for="t in ty" :key="`gy${t}`" :x1="x0" :x2="x1" :y1="y(t)" :y2="y(t)" />
      <template v-if="rejillaX"><line v-for="t in tx" :key="`gx${t}`" :y1="y0" :y2="y1" :x1="x(t)" :x2="x(t)" /></template>
    </g>
    <g class="eje">
      <line :x1="x0" :x2="x1" :y1="y0" :y2="y0" />
      <g v-for="t in tx" :key="`tx${t}`">
        <line :x1="x(t)" :x2="x(t)" :y1="y0" :y2="y0 + 6" />
        <text :x="x(t)" :y="y0 + 28" text-anchor="middle">{{ fx(t) }}</text>
      </g>
      <text v-for="t in ty" :key="`ty${t}`" :x="x0 - 12" :y="y(t) + 6" text-anchor="end">{{ fy(t) }}</text>
      <text v-if="xTitulo" class="titulo-eje" :x="(x0 + x1) / 2" :y="y0 + 60" text-anchor="middle">{{ xTitulo }}</text>
      <text v-if="yTitulo" class="titulo-eje" :x="x0" :y="y1 - 22" text-anchor="start">{{ yTitulo }}</text>
    </g>
  </g>
</template>
