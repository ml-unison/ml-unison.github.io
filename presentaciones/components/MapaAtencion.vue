<script setup lang="ts">
// Pesos de atención α_ts de una cabeza del transformer sobre las últimas 40 letras de una frase de validación.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u06-04.json'
const A = (datos as any).atencion
const props = withDefaults(defineProps<{ bloque?: number, cabeza?: number, botones?: boolean }>(), { bloque: 2, cabeza: 2, botones: true })
const sel = ref<[number, number]>([props.bloque, props.cabeza])
const letras = [...A.letras].map((c: string) => c === ' ' ? '·' : c === '\n' ? '¶' : c)
const n = letras.length, c = 11.2, m = { l: 26, t: 8 }
const W = m.l + n * c + 6, H = m.t + n * c + 26
const escalera = Array.from({ length: n - 1 }, () => `v ${c} h ${c}`).join(' ')
const color = scaleLinear<string>().domain([0, 0.15, 0.5, 1]).range(['#F1F3F8', '#C0CEE3', '#24398A', '#20305A']).clamp(true)
const celdas = computed(() => {
  const P = A.pesos[sel.value[0] - 1][sel.value[1] - 1]
  const out: { i: number, j: number, v: number }[] = []
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) if (P[i][j] > 0.01) out.push({ i, j, v: P[i][j] })
  return out
})
const cabezas = [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]] as [number, number][]
</script>

<template>
  <div class="ma">
    <div v-if="botones" class="ma__botones">
      <button v-for="h in cabezas" :key="h.join()" :class="{ activo: h[0] === sel[0] && h[1] === sel[1] }" @click="sel = h">{{ h[0] }}.{{ h[1] }}</button>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <rect :x="m.l" :y="m.t" :width="n * c" :height="n * c" fill="#FFFFFF" stroke="#E2E5EC" stroke-width="0.6" />
      <path :d="`M ${m.l} ${m.t} h ${c} ${escalera} V ${m.t + n * c} H ${m.l} Z`" fill="#F1F3F8" />
      <rect v-for="q in celdas" :key="`${q.i}-${q.j}`" :x="m.l + q.j * c" :y="m.t + q.i * c" :width="c" :height="c" :fill="color(q.v)" />
      <text v-for="(l, i) in letras" :key="`f${i}`" :x="m.l - 5" :y="m.t + i * c + c * 0.82" text-anchor="end" style="font-size: 10.5px; fill: #1A2133; font-family: var(--f-codigo)">{{ l }}</text>
      <text v-for="(l, j) in letras" :key="`c${j}`" :x="m.l + j * c + c / 2" :y="m.t + n * c + 13" text-anchor="middle" style="font-size: 10.5px; fill: #1A2133; font-family: var(--f-codigo)">{{ l }}</text>
      <text :x="m.l + n * c / 2" :y="H - 1" text-anchor="middle" style="font-size: 10px">letra atendida s →</text>
    </svg>
  </div>
</template>

<style scoped>
.ma__botones { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; }
.ma__botones button { font: 600 17px var(--f-texto); padding: 3px 11px; border-radius: 16px; border: 2px solid var(--azul); background: #fff; color: var(--azul); cursor: pointer; }
.ma__botones button.activo { background: var(--azul); color: #fff; }
</style>
