<script setup lang="ts">
// Largo de la aleta: histograma, las dos componentes de una mezcla gaussiana y su suma.
// Con `responsabilidad`, un deslizador elige un pingüino hipotético y muestra P(componente | x).
import { computed, ref } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u02.json'
const A = (datos as any).aleta
const props = withDefaults(defineProps<{ responsabilidad?: boolean, componentes?: boolean }>(), { responsabilidad: false, componentes: true })
const W = 1100, H = 360, m = { l: 20, r: 20, t: 20, b: 64 }
const x = scaleLinear().domain([170, 236]).range([m.l, W - m.r])
const ancho = 2
const cajones = computed(() => {
  const c: Record<number, number> = {}
  for (const v of A.datos) { const b = Math.floor(v / ancho) * ancho; c[b] = (c[b] ?? 0) + 1 }
  return Object.entries(c).map(([b, n]) => ({ b: +b, d: n / (A.datos.length * ancho) }))
})
const y = scaleLinear().domain([0, 0.062]).range([H - m.b, m.t])
const total = A.xs.map((_: number, i: number) => A.comp[0][i] + A.comp[1][i])
const curva = (v: number[]) => line<number>().x((_, i) => x(A.xs[i])).y(d => y(d))(v)!
const xp = ref(203)
const dens = (j: number, v: number) => A.pi[j] * Math.exp(-((v - A.mu[j]) ** 2) / (2 * A.sd[j] ** 2)) / (A.sd[j] * Math.sqrt(2 * Math.PI))
const resp = computed(() => { const a = dens(0, xp.value), b = dens(1, xp.value); return [a / (a + b), b / (a + b)] })
const colores = ['#24398A', '#C9971C']
</script>

<template>
  <div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <Ejes :x="x" :y="y" :x-ticks="[170, 180, 190, 200, 210, 220, 230]" :y-ticks="[]" x-titulo="largo de la aleta (mm)" />
      <rect v-for="c in cajones" :key="c.b" :x="x(c.b) + 1" :y="y(c.d)" :width="x(c.b + ancho) - x(c.b) - 2" :height="y(0) - y(c.d)" fill="#C0CEE3" />
      <template v-if="componentes">
        <path v-for="j in 2" :key="j" :d="curva(A.comp[j - 1])" fill="none" :stroke="colores[j - 1]" stroke-width="3" stroke-dasharray="8 5" />
        <path :d="curva(total)" fill="none" stroke="#1A2133" stroke-width="3" />
        <text :x="x(A.mu[0]) - 150" :y="y(0.058)" class="etiqueta" style="fill: #1A2133">π₁ = {{ A.pi[0] }}, μ₁ = {{ A.mu[0] }}, σ₁ = {{ A.sd[0] }}</text>
        <text :x="x(A.mu[1]) + 30" :y="y(0.036)" class="etiqueta" style="fill: #1A2133">π₂ = {{ A.pi[1] }}, μ₂ = {{ A.mu[1] }}, σ₂ = {{ A.sd[1] }}</text>
      </template>
      <template v-if="responsabilidad">
        <line :x1="x(xp)" :x2="x(xp)" :y1="y(0)" :y2="m.t" stroke="#D5433C" stroke-width="2.5" />
        <circle :cx="x(xp)" :cy="y(0)" r="8" fill="#D5433C" stroke="#fff" stroke-width="2" />
      </template>
    </svg>
    <div v-if="responsabilidad" class="m1">
      <div class="deslizador"><span>Aleta de</span><input v-model.number="xp" type="range" min="172" max="234" step="1" /><b>{{ xp }} mm</b></div>
      <div class="m1__barra">
        <div :style="{ width: `${resp[0] * 100}%`, background: colores[0] }" />
        <div :style="{ width: `${resp[1] * 100}%`, background: colores[1] }" />
      </div>
      <div class="m1__txt">P(componente 1 | x) = <b>{{ resp[0].toFixed(2) }}</b> · P(componente 2 | x) = <b>{{ resp[1].toFixed(2) }}</b></div>
    </div>
  </div>
</template>

<style scoped>
.m1 { display: flex; flex-direction: column; gap: 10px; }
.m1__barra { display: flex; height: 22px; border-radius: 4px; overflow: hidden; gap: 2px; background: #fff; }
.m1__txt { font-size: 20px; }
.m1 .deslizador input[type=range] { flex: 1 1 auto; }
.m1__txt b { color: var(--azul); }
</style>
