<script setup lang="ts">
import datos from '../datos/u01.json'
const P = (datos as any).presas
withDefaults(defineProps<{ ancho?: number, alto?: number }>(), { ancho: 660, alto: 400 })
const meses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
</script>

<template>
  <GraficaLineas :x="meses" :series="P.anual" :y-dominio="[30, 90]" :y-ticks="[30, 50, 70, 90]" :x-ticks="meses"
    :x-etiquetas="['E', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']"
    y-titulo="% del NAMO, promedio por mes del año" marcadores :ancho="ancho" :alto="alto" />
</template>
