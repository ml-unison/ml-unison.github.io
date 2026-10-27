<script setup lang="ts">
// XOR con una red 2-2-2 (tanh): el espacio original con las rectas z₁ = 0 y z₂ = 0 de las neuronas ocultas,
// y el espacio oculto (a₁, a₂), donde una recta separa las clases. Datos: herramientas/laminas/u06_01.py.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u06-01.json'
withDefaults(defineProps<{ oculto?: boolean }>(), { oculto: true })
const D = (datos as any).xor
const T = 360, m = 30
const x = scaleLinear().domain([-2.2, 2.2]).range([m, T - 6]), y = scaleLinear().domain([-2.2, 2.2]).range([T - m, 6])
const xa = scaleLinear().domain([-1.15, 1.15]).range([m, T - 6]), ya = scaleLinear().domain([-1.15, 1.15]).range([T - m, 6])
const col = (g: number) => g ? '#D5433C' : '#24398A'
// recta w·x + b = 0 recortada al cuadrado [lo, hi]²
function recta(w: number[], b: number, lo: number, hi: number) {
  const ps: number[][] = []
  for (const xv of [lo, hi]) { const yv = -(b + w[0] * xv) / w[1]; if (yv >= lo && yv <= hi) ps.push([xv, yv]) }
  for (const yv of [lo, hi]) { const xv = -(b + w[1] * yv) / w[0]; if (xv >= lo && xv <= hi) ps.push([xv, yv]) }
  return ps.slice(0, 2)
}
const ocultas = [0, 1].map(j => recta(D.W1[j], D.b1[j], -2.2, 2.2))
const wd = [D.W2[1][0] - D.W2[0][0], D.W2[1][1] - D.W2[0][1]], bd = D.b2[1] - D.b2[0]
const sep = recta(wd, bd, -1.15, 1.15)
</script>

<template>
  <div class="xco">
    <div>
      <svg class="grafica" :viewBox="`0 0 ${T} ${T}`">
        <Regiones :datos="D.regiones" :x="x" :y="y" :colores="['#C0CEE3', '#F3C7C4']" :opacidad="0.7" />
        <line v-for="(s, j) in ocultas" :key="j" :x1="x(s[0][0])" :y1="y(s[0][1])" :x2="x(s[1][0])" :y2="y(s[1][1])" stroke="#1A2133" stroke-width="2" stroke-dasharray="7 6" />
        <circle v-for="(v, i) in D.x" :key="i" :cx="x(v)" :cy="y(D.y[i])" r="4" :fill="col(D.g[i])" stroke="#fff" stroke-width="0.8" />
        <text :x="x(-0.55)" :y="y(-2.0)" style="fill: #1A2133; font-weight: 600">z₁ = 0</text>
        <text :x="x(1.05)" :y="y(1.92)" style="fill: #1A2133; font-weight: 600">z₂ = 0</text>
        <text :x="T / 2" :y="T - 6" text-anchor="middle" class="titulo-eje">x₁</text>
        <text :x="12" :y="T / 2" text-anchor="middle" class="titulo-eje">x₂</text>
      </svg>
      <div class="xco__t">espacio original (x₁, x₂)</div>
    </div>
    <div class="xco__flecha">⟶</div>
    <div :style="{ opacity: oculto ? 1 : 0.12 }">
      <svg class="grafica" :viewBox="`0 0 ${T} ${T}`">
        <rect :x="xa(-1)" :y="ya(1)" :width="xa(1) - xa(-1)" :height="ya(-1) - ya(1)" fill="#F6F7FA" stroke="#E2E5EC" />
        <line :x1="xa(sep[0][0])" :y1="ya(sep[0][1])" :x2="xa(sep[1][0])" :y2="ya(sep[1][1])" stroke="#C9971C" stroke-width="3.5" />
        <circle v-for="(v, i) in D.a1" :key="i" :cx="xa(v)" :cy="ya(D.a2[i])" r="4.5" :fill="col(D.g[i])" opacity="0.75" stroke="#fff" stroke-width="0.6" />
        <text :x="xa(-1)" :y="ya(-1) + 24" text-anchor="middle">−1</text>
        <text :x="xa(1)" :y="ya(-1) + 24" text-anchor="middle">1</text>
        <text :x="xa(-1) - 10" :y="ya(1) + 6" text-anchor="end">1</text>
        <text :x="xa(-1) - 10" :y="ya(-1) + 6" text-anchor="end">−1</text>
        <text :x="T / 2" :y="T - 6" text-anchor="middle" class="titulo-eje">a₁ = tanh(z₁)</text>
        <text :x="12" :y="T / 2" text-anchor="middle" class="titulo-eje">a₂</text>
      </svg>
      <div class="xco__t">espacio oculto (a₁, a₂)</div>
    </div>
  </div>
</template>

<style scoped>
.xco { display: grid; grid-template-columns: 1fr 30px 1fr; gap: 8px; align-items: center; }
.xco__t { text-align: center; font-weight: 600; font-size: 19px; color: var(--marino); margin-top: 2px; }
.xco__flecha { font-size: 34px; color: var(--suave); text-align: center; }
</style>
