<script setup lang="ts">
// Descenso por gradiente en (w0, w1) para total ~ temp: con la temperatura cruda y estandarizada.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u04.json'
const D = (datos as any).descenso
const W = 520, H = 380, m = { l: 64, r: 16, t: 40, b: 64 }
const panel = (k: string, dom: number[]) => {
  const d = D[k]
  const x = scaleLinear().domain([dom[0], dom[1]]).range([m.l, W - m.r]), y = scaleLinear().domain([dom[2], dom[3]]).range([H - m.b, m.t])
  const l = line<number[]>().x(p => x(p[0])).y(p => y(p[1]))
  return { d, x, y, elipses: d.elipses.map((e: number[][]) => l(e)), camino: l(d.camino) }
}
const crudo = panel('crudo', [-0.5, 7, 0, 18]), estandar = panel('estandar', [-20, 260, -40, 190])
const props = withDefaults(defineProps<{ ambos?: boolean }>(), { ambos: true })
</script>

<template>
  <div class="dv">
    <div v-for="(p, k) in (ambos ? [crudo, estandar] : [crudo])" :key="k">
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <clipPath :id="`dv${k}`"><rect :x="m.l" :y="m.t" :width="W - m.l - m.r" :height="H - m.t - m.b" /></clipPath>
        <Ejes :x="p.x" :y="p.y" :x-ticks="4" :y-ticks="4" x-titulo="w₀ (intersección)" y-titulo="w₁ (pendiente)" />
        <g :clip-path="`url(#dv${k})`">
          <path v-for="(e, i) in p.elipses" :key="i" :d="e" fill="none" stroke="#8DA3C9" stroke-width="1.6" />
          <path :d="p.camino" fill="none" stroke="#D5433C" stroke-width="2.5" />
          <circle v-for="(q, i) in p.d.camino.slice(0, 4)" :key="`c${i}`" :cx="p.x(q[0])" :cy="p.y(q[1])" r="4.5" fill="#D5433C" />
          <circle :cx="p.x(p.d.opt[0])" :cy="p.y(p.d.opt[1])" r="7" fill="#C9971C" stroke="#fff" stroke-width="2" />
        </g>
      </svg>
      <div class="dv__t">{{ k === 0 ? 'temperatura en °C' : 'temperatura estandarizada' }} · κ = {{ p.d.kappa.toLocaleString('es-MX') }} · {{ p.d.iter.toLocaleString('es-MX') }} {{ p.d.iter === 1 ? 'iteración' : 'iteraciones' }}</div>
    </div>
  </div>
</template>

<style scoped>
.dv { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
.dv__t { text-align: center; font-weight: 600; font-size: 19px; color: var(--marino); margin-top: -4px; }
</style>
