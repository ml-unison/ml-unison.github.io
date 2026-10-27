<script setup lang="ts">
import datos from '../datos/u01.json'
const C = (datos as any).casas
</script>

<template>
  <div class="cols" style="gap: 40px">
    <Histograma :cuentas="C.hist_precio.cuentas" :bordes="C.hist_precio.bordes" titulo="Precio" x-titulo="miles de dólares" :x-ticks="[0, 200, 400, 600]" />
    <Histograma :cuentas="C.hist_log.cuentas" :bordes="C.hist_log.bordes" titulo="Logaritmo del precio" x-titulo="log₁₀(dólares)" :x-ticks="[4.6, 5, 5.4, 5.8]" />
  </div>
</template>
