<script setup lang="ts">
import datos from '../datos/u01.json'
const P = (datos as any).presas
const orden = Object.fromEntries(Object.entries(P.promedio as Record<string, number>).reverse())
</script>

<template>
  <BarrasHorizontales :datos="orden" :resaltar="['Los Pilares']" unidad=" %" :max="100" :notas="{ 'Los Pilares': 'solo 3 meses' }" />
</template>
