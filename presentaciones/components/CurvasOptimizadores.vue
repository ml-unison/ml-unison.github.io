<script setup lang="ts">
// Pérdida por época en entrenamiento y validación para varias corridas del MLP en Fashion-MNIST.
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-02.json'
const C = (datos as any).corridas as any[]
const props = withDefaults(defineProps<{ corridas?: string[], colores?: string[], etiquetas?: string[], ymax?: number, alto?: number }>(), {
  corridas: () => ['SGD', 'SGD + momento', 'Adam'], colores: () => ['#6B7385', '#24398A', '#D5433C'], ymax: 1.0, alto: 400,
})
const W = 540, H = props.alto, m = { l: 56, r: 140, t: 40, b: 66 }
const x = scaleLinear().domain([1, 30]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, props.ymax]).range([H - m.b, m.t]).clamp(true)
const sel = props.corridas.map((n, k) => ({ n, c: props.colores[k], et: props.etiquetas?.[k] ?? n, h: C.find(c => c.nombre === n) }))
const d = (v: number[]) => line<number>().x((_, i) => x(i + 1)).y(w => y(w))(v)!
const paneles = [{ k: 'pe', titulo: 'pérdida en entrenamiento' }, { k: 'pv', titulo: 'pérdida en validación' }].map(p => {
  const et = sel.map(s => ({ ...s, y: y(s.h[p.k][s.h[p.k].length - 1]) })).sort((a, b) => a.y - b.y)
  for (let i = 1; i < et.length; i++) et[i].y = Math.max(et[i].y, et[i - 1].y + 24)
  return { ...p, et }
})
</script>

<template>
  <div class="co">
    <svg v-for="p in paneles" :key="p.k" class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <text :x="m.l" :y="18" style="fill: #20305A; font-weight: 650; font-size: 20px">{{ p.titulo }}</text>
      <Ejes :x="x" :y="y" :x-ticks="[1, 10, 20, 30]" :y-ticks="[0, 0.2, 0.4, 0.6, 0.8, 1.0].filter(v => v <= ymax)" x-titulo="época" />
      <path v-for="s in sel" :key="s.n" :d="d(s.h[p.k])" fill="none" :stroke="s.c" stroke-width="3" stroke-linejoin="round" />
      <g v-for="e in p.et" :key="`e${e.n}`">
        <line :x1="W - m.r + 8" :x2="W - m.r + 24" :y1="e.y" :y2="e.y" :stroke="e.c" stroke-width="3" />
        <text :x="W - m.r + 30" :y="e.y + 6" style="fill: #1A2133; font-weight: 600; font-size: 18px">{{ e.et }}</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.co { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
</style>
