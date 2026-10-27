<script setup lang="ts">
// Los 1,797 dígitos proyectados en 2D (PCA, t-SNE o UMAP), con el número en la mediana de cada grupo.
import { computed } from 'vue'
import datos from '../datos/u02.json'
import { DIEZ } from './colores'
const G = (datos as any).digitos
const props = withDefaults(defineProps<{ metodo: string, ancho?: number, alto?: number, titulo?: string, radio?: number }>(), { ancho: 560, alto: 460, radio: 3 })
const pts = computed(() => props.metodo.startsWith('perp') ? G.perp[props.metodo.slice(4)] : G[props.metodo])
const y = computed(() => props.metodo.startsWith('perp') ? G.sub_y : G.y)
const med = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
const etiquetas = computed(() => Array.from({ length: 10 }, (_, d) => {
  const idx = y.value.map((c: number, i: number) => c === d ? i : -1).filter((i: number) => i >= 0)
  return { x: med(idx.map((i: number) => pts.value[i][0])), y: med(idx.map((i: number) => pts.value[i][1])), texto: String(d) }
}))
</script>

<template>
  <Dispersion :x="pts.map((p: number[]) => p[0])" :y="pts.map((p: number[]) => p[1])" :g="y" :colores="DIEZ" :ancho="ancho" :alto="alto" :radio="radio" :opacidad="0.6" :titulo="titulo">
    <template #default="{ x, y: yy }">
      <g v-for="e in etiquetas" :key="e.texto">
        <circle :cx="x(e.x)" :cy="yy(e.y)" r="15" fill="#fff" opacity="0.9" :stroke="DIEZ[+e.texto]" stroke-width="2.5" />
        <text :x="x(e.x)" :y="yy(e.y) + 7" text-anchor="middle" style="fill: #1A2133; font-weight: 700; font-size: 19px">{{ e.texto }}</text>
      </g>
    </template>
  </Dispersion>
</template>
