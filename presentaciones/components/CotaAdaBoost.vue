<script setup lang="ts">
// AdaBoost en los pingüinos: error de entrenamiento contra la cota ∏ 2√(ε(1−ε)) (= pérdida exponencial media).
import datos from '../datos/u05-03.json'
const C = (datos as any).cota
withDefaults(defineProps<{ ancho?: number, alto?: number }>(), { ancho: 640, alto: 400 })
</script>

<template>
  <LineasLog :x="C.t" :series="{ 'cota ∏ 2√(ε(1−ε))': C.cota, 'error de entrenamiento': C.error }" :y-log="true"
             :y-dominio="[0.02, 1]" :y-ticks="[0.02, 0.05, 0.1, 0.2, 0.5, 1]" :x-dominio="[0, 200]" :x-ticks="[0, 50, 100, 150, 200]"
             :colores="['#C9971C', '#24398A']" x-titulo="ronda t" y-titulo="proporción (escala logarítmica)" :ancho="ancho" :alto="alto" :derecha="220" />
</template>
