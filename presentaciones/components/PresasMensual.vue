<script setup lang="ts">
import datos from '../datos/u01.json'
const P = (datos as any).presas
</script>

<template>
  <GraficaLineas :x="P.mensual.fechas" :series="P.mensual.series" fechas :y-dominio="[0, 125]" :y-ticks="[0, 25, 50, 75, 100, 125]"
    y-titulo="% del NAMO, promedio mensual" :referencia="{ y: 100, texto: 'NAMO' }" :alto="380"
    :x-ticks="['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']" />
</template>
