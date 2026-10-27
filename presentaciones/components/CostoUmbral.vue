<script setup lang="ts">
// Costo promedio por cliente contra el umbral (c_FP = 1, c_FN = 5), con t* = 1/6 marcado.
import datos from '../datos/u03.json'
const K = (datos as any).costos
</script>

<template>
  <LineasLog :x="K.t" :series="{ 'bosque': K.bosque, 'logística': K.logistica }" :y-log="false" :y-dominio="[0.4, 1.2]" :y-ticks="[0.4, 0.6, 0.8, 1, 1.2]"
             :x-dominio="[0, 0.9]" :x-ticks="[0, 0.2, 0.4, 0.6, 0.8]" x-titulo="umbral t" y-titulo="costo promedio por cliente" :colores="['#24398A', '#D5433C']" :ancho="720" :alto="400" :derecha="120">
    <template #default="{ x, y }">
      <line :x1="x(K.t_bayes)" :x2="x(K.t_bayes)" :y1="y(0.4)" :y2="y(1.2)" stroke="#C9971C" stroke-width="2" stroke-dasharray="5 4" />
      <text :x="x(K.t_bayes) + 8" :y="y(1.15)" style="fill: #1A2133; font-weight: 600">t* = 1/6</text>
      <line :x1="x(0.5)" :x2="x(0.5)" :y1="y(0.4)" :y2="y(1.2)" stroke="#8A91A0" stroke-width="1.5" stroke-dasharray="3 4" />
      <text :x="x(0.5) + 8" :y="y(1.15)" style="font-size: 16px">t = 0.5</text>
    </template>
  </LineasLog>
</template>
