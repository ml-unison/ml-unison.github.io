<script setup lang="ts">
// Inercia (codo) y silueta media para k = 2…8 en los 4,317 clientes.
import datos from '../datos/u02.json'
const C = (datos as any).clientes
</script>

<template>
  <div class="ck">
    <div><div class="ck__t">Inercia J (método del codo)</div>
      <GraficaLineas :x="C.k" :series="{ 'J': C.inercia }" :y-dominio="[0, 7000]" :y-ticks="[0, 2000, 4000, 6000]" :ancho="540" :alto="300" marcadores sin-etiquetas /></div>
    <div><div class="ck__t">Silueta media</div>
      <GraficaLineas :x="C.k" :series="{ 's': C.silueta }" :y-dominio="[0.25, 0.45]" :y-ticks="[0.25, 0.3, 0.35, 0.4, 0.45]" :ancho="540" :alto="300" marcadores sin-etiquetas /></div>
  </div>
</template>

<style scoped>
.ck { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
.ck__t { font-weight: 600; font-size: 21px; color: var(--marino); margin-bottom: 4px; }
</style>
