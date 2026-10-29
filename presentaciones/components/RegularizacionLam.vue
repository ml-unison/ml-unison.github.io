<script setup lang="ts">
// Sesgo² y varianza del polinomio de grado 12 regularizado, contra λ.
import datos from '../datos/u03.json'
const R = (datos as any).regularizacion
// desde λ = 10⁻⁴: con λ menor la varianza es tan grande que la estimación del sesgo es puro ruido
const i0 = R.lam.findIndex((l: number) => l >= 9e-5)
const lam = R.lam.slice(i0)
const s = { 'sesgo²': R.sesgo2.slice(i0), 'varianza': R.var.slice(i0), 'suma': R.sesgo2.slice(i0).map((v: number, i: number) => v + R.var[i0 + i]) }
const sup = (e: number) => `10${String(e).replace('-', '⁻').replace(/\d/g, (d: string) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d])}`
</script>

<template>
  <LineasLog :x="lam" :series="s" x-log :y-dominio="[0.001, 1]" :y-ticks="[0.001, 0.01, 0.1, 1]" :x-ticks="[1e-4, 1e-3, 1e-2, 1e-1, 1, 10]" :formato-x="(v: number) => sup(Math.round(Math.log10(v)))"
             x-titulo="λ (escala log)" y-titulo="error cuadrático medio (escala log)" :colores="['#24398A', '#D5433C', '#1A2133']" :punteadas="['varianza']" marcadores :ancho="720" :alto="330" :derecha="120" />
</template>
