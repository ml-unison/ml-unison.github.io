<script setup lang="ts">
// RECM de prueba en las bicicletas: la escalera lineal y Poisson de la Unidad 4 contra los árboles.
import datos from '../datos/u05-01.json'
withDefaults(defineProps<{ ancho?: number }>(), { ancho: 560 })
const R = (datos as any).resumen as [string, number][]
const filas = Object.fromEntries(R.filter(([n]) => !n.startsWith('M1') && !n.startsWith('M2')).map(([n, v]) => [n.replace(' (U4)', ''), v]))
</script>

<template>
  <BarrasHorizontales :datos="filas" :resaltar="['árbol podado']" :max="140" :ancho="ancho" :izquierda="290" />
</template>
