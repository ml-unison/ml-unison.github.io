<script setup lang="ts">
// Un día laboral de prueba, hora por hora: observado, regresión lineal (M4) y regresión de Poisson (M4).
import datos from '../datos/u04.json'
const D = (datos as any).dia
</script>

<template>
  <LineasLog :x="D.hora" :series="{ 'observado': D.real, 'lineal': D.lineal, 'Poisson': D.poisson }" :y-log="false" :y-dominio="[-100, 500]" :y-ticks="[-100, 0, 100, 200, 300, 400, 500]"
             :x-dominio="[0, 23]" :x-ticks="[0, 4, 8, 12, 16, 20]" :colores="['#1A2133', '#D5433C', '#24398A']" marcadores
             x-titulo="hora" :y-titulo="`bicicletas por hora · ${D.fecha}`" :ancho="760" :alto="400" :derecha="140">
    <template #default="{ x, y }"><line :x1="x(0)" :x2="x(23)" :y1="y(0)" :y2="y(0)" stroke="#1A2133" stroke-width="1.2" /></template>
  </LineasLog>
</template>
