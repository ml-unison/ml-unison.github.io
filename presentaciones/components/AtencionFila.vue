<script setup lang="ts">
// Una fila de la matriz de atención: a qué letras anteriores mira la letra seleccionada (clic para cambiarla).
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u06-04.json'
const A = (datos as any).atencion
const props = withDefaults(defineProps<{ bloque?: number, cabeza?: number, consulta?: number }>(), { bloque: 2, cabeza: 2, consulta: -1 })
const letras = [...A.letras].map((c: string) => c === ' ' ? '·' : c === '\n' ? '¶' : c)
const n = letras.length
const t = ref(props.consulta < 0 ? n + props.consulta : props.consulta)
const P = A.pesos[props.bloque - 1][props.cabeza - 1]
const color = scaleLinear<string>().domain([0, 0.15, 0.5, 1]).range(['#FFFFFF', '#C0CEE3', '#24398A', '#20305A']).clamp(true)
const fila = computed(() => P[t.value] as number[])
</script>

<template>
  <div>
    <div class="af">
      <span v-for="(l, j) in letras" :key="j" @click="t = j"
        :class="{ consulta: j === t, futuro: j > t }"
        :style="{ background: j <= t ? color(fila[j]) : 'transparent', color: j <= t && fila[j] > 0.35 ? '#fff' : '#1A2133' }">{{ l }}</span>
    </div>
    <div class="af__pie">
      consulta: «<b>{{ letras[t] }}</b>» (recuadro rojo) · mayor peso: «{{ letras[fila.indexOf(Math.max(...fila))] }}», α = {{ Math.max(...fila).toFixed(2) }} · haz clic en otra letra
    </div>
  </div>
</template>

<style scoped>
.af { display: flex; flex-wrap: wrap; row-gap: 6px; font-family: var(--f-codigo); font-size: 26px; line-height: 1; }
.af span { display: inline-block; width: 0.8em; padding: 9px 0; text-align: center; cursor: pointer; border: 2px solid transparent; }
.af span.consulta { border-color: #D5433C; }
.af span.futuro { color: #C5C9D2 !important; }
.af__pie { margin-top: 12px; font-size: 18px; color: var(--tenue); }
.af__pie b { color: var(--rojo); }
</style>
