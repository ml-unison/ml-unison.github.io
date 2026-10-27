<script setup lang="ts">
// BIC contra número de componentes (4 medidas estandarizadas), por tipo de covarianza.
import datos from '../datos/u02.json'
const B = (datos as any).bic
const series = { 'esférica': B.bic.spherical, 'diagonal': B.bic.diag, 'compartida': B.bic.tied, 'completa': B.bic.full }
</script>

<template>
  <GraficaLineas :x="B.k" :series="series" :y-dominio="[2400, 4000]" :y-ticks="[2500, 3000, 3500, 4000]" :ancho="760" :alto="400" marcadores y-titulo="BIC (menor es mejor)" />
</template>
