<script setup lang="ts">
// Función de crecimiento: 2^N contra los conjuntos con punto de quiebre (polinomiales).
import datos from '../datos/u03.json'
const C = (datos as any).crecimiento
const s = { '2ᴺ': C.todas, 'perceptrón en ℝ²': C.perceptron, 'intervalos': C.intervalos, 'rayos': C.rayos }
</script>

<template>
  <LineasLog :x="C.N" :series="s" :y-dominio="[1, 5000]" :y-ticks="[1, 10, 100, 1000]" :x-ticks="[1, 3, 6, 9, 12]" x-titulo="N" y-titulo="m_H(N)"
             :colores="['#6B7385', '#24398A', '#C9971C', '#D5433C']" :punteadas="['2ᴺ']" marcadores :ancho="720" :alto="420" :derecha="190" />
</template>
