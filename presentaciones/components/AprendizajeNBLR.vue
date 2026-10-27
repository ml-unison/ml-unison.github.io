<script setup lang="ts">
// Error de prueba contra tamaño de entrenamiento en los SMS: Bayes ingenuo contra regresión logística.
import datos from '../datos/u04.json'
const A = (datos as any).aprendizaje
</script>

<template>
  <LineasLog :x="A.n" :series="{ 'Bayes ingenuo': A.nb, 'logística': A.logistica }" x-log :y-dominio="[0.01, 0.2]" :y-ticks="[0.01, 0.02, 0.05, 0.1, 0.2]"
             :x-ticks="[20, 50, 100, 200, 500, 1000, 2000, 4000]" x-titulo="mensajes de entrenamiento (escala log)" y-titulo="error de prueba (escala log)"
             :colores="['#D5433C', '#24398A']" :ancho="720" :alto="400" :derecha="170" marcadores />
</template>
