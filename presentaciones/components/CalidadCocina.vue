<script setup lang="ts">
import datos from '../datos/u01.json'
const C = (datos as any).casas
</script>

<template>
  <GraficaLineas :x="[0, 1, 2, 3]" :series="{ 'precio mediano': C.cocina.mediana }" :x-ticks="[0, 1, 2, 3]"
    :x-etiquetas="['Regular (Fa)', 'Típica (TA)', 'Buena (Gd)', 'Excelente (Ex)']" :y-dominio="[100, 330]" :y-ticks="[100, 150, 200, 250, 300]"
    y-titulo="miles de dólares" marcadores :ancho="640" :alto="380" />
</template>
