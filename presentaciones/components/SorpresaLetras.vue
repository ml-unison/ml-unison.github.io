<script setup lang="ts">
// Sorpresa −log₂ P(letra | contexto) del transformer en cada letra de una frase de validación.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u06-04.json'
const S = (datos as any).sorpresa
const color = scaleLinear<string>().domain([0, 2.5, 5.5, 8.5]).range(['#FFFFFF', '#C0CEE3', '#24398A', '#20305A']).clamp(true)
// palabras con su espacio final, para no partir una palabra al cambiar de renglón
const palabras: { c: string, b: number }[][] = [[]]
;[...S.frase].forEach((c: string, i: number) => {
  palabras[palabras.length - 1].push({ c: c === ' ' ? '·' : c, b: S.bits[i] })
  if (c === ' ') palabras.push([])
})
const total = S.bits.reduce((a: number, b: number) => a + b, 0)
</script>

<template>
  <div>
    <div class="sp">
      <span v-for="(p, k) in palabras" :key="k" class="sp__palabra">
        <span v-for="(l, i) in p" :key="i" :style="{ background: color(l.b), color: l.b > 4.5 ? '#fff' : '#1A2133' }" :title="`${l.b.toFixed(1)} bits`">{{ l.c }}</span>
      </span>
    </div>
    <div class="sp__escala">
      <span>0 bits</span>
      <i v-for="k in 9" :key="k" :style="{ background: color(k - 1) }" />
      <span>8 bits</span>
      <span class="sp__total">promedio en la frase: {{ (total / S.bits.length).toFixed(2) }} bits por letra</span>
    </div>
  </div>
</template>

<style scoped>
.sp { display: flex; flex-wrap: wrap; row-gap: 8px; font-family: var(--f-codigo); font-size: 27px; line-height: 1; }
.sp__palabra { display: inline-flex; }
.sp__palabra span { display: inline-block; width: 0.78em; padding: 9px 0; text-align: center; }
.sp__escala { display: flex; align-items: center; gap: 0; margin-top: 18px; font-size: 18px; color: var(--tenue); }
.sp__escala i { display: inline-block; width: 26px; height: 16px; border: 0.5px solid #E2E5EC; }
.sp__escala span { margin: 0 10px; }
.sp__total { margin-left: auto !important; color: var(--tinta); }
</style>
