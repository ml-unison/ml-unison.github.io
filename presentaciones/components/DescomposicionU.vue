<script setup lang="ts">
// Sesgo², varianza y su suma contra el grado del polinomio (N = 15, ruido σ² = 0.0625).
import datos from '../datos/u03.json'
const S = (datos as any).sesgo_varianza
const g = S.grado.slice(0, 11)
const s = { 'sesgo²': S.sesgo2.slice(0, 11), 'varianza': S.var.slice(0, 11), 'sesgo² + varianza': S.sesgo2.slice(0, 11).map((v: number, i: number) => v + S.var[i]) }
</script>

<template>
  <LineasLog :x="g" :series="s" :y-dominio="[0.00005, 5]" :y-ticks="[0.0001, 0.001, 0.01, 0.1, 1]" :x-ticks="[0, 2, 4, 6, 8, 10]" x-titulo="grado del polinomio" y-titulo="error cuadrático medio (escala log)"
             :colores="['#24398A', '#D5433C', '#1A2133']" :punteadas="['varianza']" marcadores :ancho="720" :alto="420" :derecha="200" />
</template>
