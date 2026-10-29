<script setup lang="ts">
// Elegir el mejor de varios modelos que adivinan (error real 0.5) por su error de validación con 100 datos.
import datos from '../datos/u03.json'
const O = (datos as any).optimismo
</script>

<template>
  <LineasLog :x="O.M" :series="{ 'E_val del elegido': O.mejor }" x-log :y-log="false" :y-dominio="[0.3, 0.55]" :y-ticks="[0.3, 0.35, 0.4, 0.45, 0.5, 0.55]"
             :x-ticks="[1, 10, 100, 500]" x-titulo="número de modelos comparados (escala log)" y-titulo="error de validación" marcadores :ancho="720" :alto="330" :derecha="190">
    <template #default="{ x, y }">
      <line :x1="x(1)" :x2="x(500)" :y1="y(0.5)" :y2="y(0.5)" stroke="#D5433C" stroke-width="2" stroke-dasharray="6 5" />
      <text :x="x(30)" :y="y(0.5) - 10" style="fill: #1A2133; font-size: 16px">error real de todos: 0.5</text>
    </template>
  </LineasLog>
</template>
