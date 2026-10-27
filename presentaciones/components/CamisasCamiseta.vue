<script setup lang="ts">
// Camisas del conjunto de prueba que la red convolucional clasificó como «camiseta».
import datos from '../datos/u06-03.json'
const D = datos as any
withDefaults(defineProps<{ tam?: number }>(), { tam: 74 })
</script>

<template>
  <div class="cc"><MapaPixeles v-for="(im, k) in D.camisas" :key="k" :datos="im" :tam="tam" :vmax="255" /></div>
</template>

<style scoped>
.cc { display: flex; gap: 8px; margin: 4px 0 18px; }
</style>
