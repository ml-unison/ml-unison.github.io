<script setup lang="ts">
// Prueba KS con un corrimiento fijo de 0.5 °C: el estadístico D se estabiliza, el valor p se desploma.
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import d from '../datos/u08-02.json'
const K = (d as any).ks_n as { n: number, D: number, log10p: number, umbral: number }[]
const W = 540, H = 320, m = { l: 62, r: 16, t: 34, b: 60 }
const x = scaleLog().domain([80, 125000]).range([m.l, W - m.r])
const yD = scaleLinear().domain([0, 0.2]).range([H - m.b, m.t])
const yP = scaleLinear().domain([-80, 0]).range([H - m.b, m.t])
const xt = [100, 1000, 10000, 100000]
const fx = (v: number) => v.toLocaleString('es-MX')
const camino = (f: (k: any) => number, y: any) => line<any>().x(k => x(k.n)).y(k => y(f(k)))(K)!
</script>

<template>
  <div class="ks">
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <Ejes :x="x" :y="yD" :x-ticks="xt" :y-ticks="[0, 0.05, 0.1, 0.15, 0.2]" :x-formato="fx" x-titulo="datos en cada muestra" y-titulo="estadístico D" />
      <path :d="camino(k => k.umbral, yD)" fill="none" stroke="#D5433C" stroke-width="2.4" stroke-dasharray="7 5" />
      <path :d="camino(k => k.D, yD)" fill="none" stroke="#24398A" stroke-width="3" />
      <circle v-for="k in K" :key="k.n" :cx="x(k.n)" :cy="yD(k.D)" r="5" fill="#24398A" stroke="#fff" stroke-width="1.5" />
      <text :x="x(130)" :y="yD(0.19)" class="etiqueta" style="fill:#D5433C">umbral al 5 %</text>
      <text :x="x(6000)" :y="yD(0.065)" class="etiqueta" style="fill:#24398A">D observado</text>
    </svg>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <Ejes :x="x" :y="yP" :x-ticks="xt" :y-ticks="[]" :x-formato="fx" x-titulo="datos en cada muestra" y-titulo="valor p (escala log)" />
      <g v-for="t in [-80, -60, -40, -20, 0]" :key="t">
        <line class="rejilla-linea" :x1="m.l" :x2="W - m.r" :y1="yP(t)" :y2="yP(t)" stroke="#E2E5EC" />
        <text :x="m.l - 10" :y="yP(t) + 6" text-anchor="end"><template v-if="t === 0">1</template><template v-else>10<tspan dy="-8" style="font-size: 13px">−{{ -t }}</tspan></template></text>
      </g>
      <line :x1="m.l" :x2="W - m.r" :y1="yP(Math.log10(0.05))" :y2="yP(Math.log10(0.05))" stroke="#D5433C" stroke-width="2.4" stroke-dasharray="7 5" />
      <path :d="camino(k => k.log10p, yP)" fill="none" stroke="#24398A" stroke-width="3" />
      <circle v-for="k in K" :key="k.n" :cx="x(k.n)" :cy="yP(k.log10p)" r="5" fill="#24398A" stroke="#fff" stroke-width="1.5" />
      <text :x="x(130)" :y="yP(-14)" class="etiqueta" style="fill:#D5433C">↑ p = 0.05</text>
    </svg>
  </div>
</template>

<style scoped>
.ks { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
</style>
