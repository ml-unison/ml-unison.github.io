<script setup lang="ts">
// Norma de w en las iteraciones de Newton con datos linealmente separables: crece sin límite.
import datos from '../datos/u04.json'
const S = (datos as any).separable as number[][]
</script>

<template>
  <LineasLog :x="S.map(p => p[0])" :series="{ '‖w‖': S.map(p => p[1]) }" :y-log="false" :y-dominio="[0, 1000]" :y-ticks="[0, 250, 500, 750, 1000]"
             :x-dominio="[0, 25]" :x-ticks="[0, 5, 10, 15, 20, 25]" x-titulo="iteración de Newton" y-titulo="‖w‖" :ancho="600" :alto="380" :derecha="80" marcadores />
</template>
