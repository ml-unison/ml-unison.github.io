<script setup lang="ts">
// Descenso con momento en un valle cuadrático con κ = 50: la misma tasa η y distintos β (botones).
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-02.json'
const V = (datos as any).valle
const props = withDefaults(defineProps<{ inicial?: number, pasos?: number }>(), { inicial: 0, pasos: 40 })
const W = 1120, H = 300, m = { l: 20, r: 20, t: 14, b: 14 }
const x = scaleLinear().domain([-4.35, 1.05]).range([m.l, W - m.r])
const esc = (x(1) - x(0))
const y = (v: number) => H / 2 - v * esc
const t = Array.from({ length: 121 }, (_, i) => i / 120 * 2 * Math.PI)
const elipses = [0.05, 0.3, 1, 2.5, 5, 8.5, 13].map(c => line<number>().x(s => x(Math.sqrt(2 * c) * Math.cos(s))).y(s => y(Math.sqrt(2 * c / V.kappa) * Math.sin(s)))(t)!)
const colores: Record<string, string> = { '0': '#D5433C', '0.5': '#6B7385', '0.64': '#24398A', '0.9': '#C9971C' }
const beta = ref(props.inicial)
const sel = computed(() => V.caminos.find((c: any) => c.beta === beta.value))
const camino = computed(() => line<number[]>().x(p => x(p[0])).y(p => y(p[1]))(sel.value.camino.slice(0, props.pasos + 1))!)
const fmt = (v: number) => String(v).replace('.', '.')
</script>

<template>
  <div>
    <div class="vm__botones">
      <span class="vm__eta">η = {{ V.eta.toFixed(3) }} fija · momento:</span>
      <button v-for="c in V.caminos" :key="c.beta" :class="{ activo: c.beta === beta }" :style="{ '--c': colores[String(c.beta)] }" @click="beta = c.beta">
        β = {{ fmt(c.beta) }}
      </button>
      <span class="vm__iter"><b :style="{ color: colores[String(beta)] }">{{ sel.iter }}</b> iteraciones hasta reducir el error a 10⁻⁶</span>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <clipPath id="vm-clip"><rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" /></clipPath>
      <rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" fill="#FAFBFD" stroke="#E2E5EC" />
      <g clip-path="url(#vm-clip)">
        <path v-for="(e, i) in elipses" :key="i" :d="e" fill="none" stroke="#8DA3C9" stroke-width="1.6" />
        <path :d="camino" fill="none" :stroke="colores[String(beta)]" stroke-width="2.6" stroke-linejoin="round" />
        <circle v-for="(p, i) in sel.camino.slice(0, pasos + 1)" :key="i" :cx="x(p[0])" :cy="y(p[1])" r="4.5" :fill="colores[String(beta)]" />
      </g>
      <path :d="`M ${x(0)} ${y(0) - 15} l 4.4 9.5 10.3 1.2 -7.6 7 2.1 10.2 -9.2 -5.2 -9.2 5.2 2.1 -10.2 -7.6 -7 10.3 -1.2 z`" fill="#C9971C" stroke="#fff" stroke-width="1.5" />
      <text :x="x(-4.3)" :y="H - 24" style="fill: #5E6678; font-size: 17px">primeros {{ pasos }} pasos · μ = 1 a lo largo del valle, μ = 50 a lo ancho</text>
    </svg>
  </div>
</template>

<style scoped>
.vm__botones { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; font-size: 21px; }
.vm__eta { color: var(--tenue); margin-right: 4px; }
.vm__botones button { font: 600 19px var(--f-texto); padding: 5px 14px; border-radius: 18px; border: 2px solid var(--c); background: #fff; color: var(--c); cursor: pointer; }
.vm__botones button.activo { background: var(--c); color: #fff; }
.vm__iter { margin-left: auto; color: var(--tinta); }
.vm__iter b { font-family: var(--f-titulo); font-size: 26px; }
</style>
