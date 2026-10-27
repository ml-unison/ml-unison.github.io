<script setup lang="ts">
// Pronósticos de referencia desde un origen fijo (15 de enero de 2023) para los 30 días siguientes, contra lo observado.
import { computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-01.json'
const props = withDefaults(defineProps<{ mostrar?: number, observado?: boolean }>(), { mostrar: 4, observado: true })
const E = (datos as any).ejemplo
const W = 1120, H = 390, m = { l: 56, r: 24, t: 40, b: 66 }
const np = E.n_pasado - 1 // el origen es el día 0
const x = scaleLinear().domain([-np, 30]).range([m.l, W - m.r]), y = scaleLinear().domain([8, 25]).range([H - m.b, m.t])
const pasado = line<number>().x((_, i) => x(i - np)).y(v => y(v))(E.pasado)!
const futuro = line<number>().x((_, i) => x(i + 1)).y(v => y(v))(E.futuro)!
const metodos = [
  { n: 'ingenuo', d: E.ingenuo, c: '#D5433C' },
  { n: 'climatología', d: E.clima, c: '#24398A' },
  { n: 'AR(1) de anomalías', d: E.ar1, c: '#C9971C' },
  { n: 'estacional ingenuo', d: E.estacional, c: '#6B7385' },
]
const caminos = computed(() => metodos.slice(0, props.mostrar).map(o => ({ ...o, p: line<number>().x((_, i) => x(i + 1)).y(v => y(v))(o.d)! })))
</script>

<template>
  <div>
    <div class="ref__leyenda">
      <span v-for="o in metodos.slice(0, mostrar)" :key="o.n"><i :style="{ background: o.c }" />{{ o.n }}</span>
      <span v-if="observado"><b>•</b> lo que pasó</span>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
      <Ejes :x="x" :y="y" :x-ticks="[-45, -30, -15, 0, 10, 20, 30]" :x-formato="v => v === 0 ? '15 ene 2023' : (v > 0 ? `+${v}` : String(v).replace('-', '−'))" :y-ticks="[10, 15, 20, 25]" y-titulo="°C" x-titulo="días respecto al origen" />
      <line :x1="x(0)" :x2="x(0)" :y1="m.t" :y2="H - m.b" stroke="#8A91A0" stroke-dasharray="6 5" />
      <text :x="x(0) + 8" :y="m.t + 14" style="fill: #1A2133; font-weight: 600">origen T</text>
      <path :d="pasado" fill="none" stroke="#1A2133" stroke-width="2.6" />
      <path v-for="o in caminos" :key="o.n" :d="o.p" fill="none" :stroke="o.c" stroke-width="3.6" />
      <template v-if="observado">
        <path :d="futuro" fill="none" stroke="#1A2133" stroke-width="1.4" stroke-dasharray="3 4" />
        <circle v-for="(v, i) in E.futuro" :key="i" :cx="x(i + 1)" :cy="y(v)" r="4.5" fill="#1A2133" />
      </template>
    </svg>
  </div>
</template>

<style scoped>
.ref__leyenda { display: flex; gap: 26px; font-size: 20px; color: var(--tinta); margin-bottom: 4px; flex-wrap: wrap; }
.ref__leyenda i { display: inline-block; width: 22px; height: 5px; border-radius: 2px; margin-right: 8px; vertical-align: 5px; }
.ref__leyenda b { font-size: 26px; line-height: 0; margin-right: 6px; }
</style>
