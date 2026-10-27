<script setup lang="ts">
// Los 16 filtros de 3 × 3 de la primera capa (azul +, rojo −) y su mapa de activación (tras la ReLU) en el botín.
import datos from '../datos/u06-03.json'
const D = datos as any
const lim = Math.max(...D.filtros.flat(2).map(Math.abs))
withDefaults(defineProps<{ tam?: number }>(), { tam: 92 })
</script>

<template>
  <div class="fa">
    <div v-for="(f, k) in D.filtros" :key="k" class="fa__par">
      <MapaPixeles :datos="f" :tam="tam * 0.62" modo="divergente" :vmax="lim" :marco="false" />
      <MapaPixeles :datos="D.mapas[k]" :tam="tam" />
    </div>
  </div>
</template>

<style scoped>
.fa { display: grid; grid-template-columns: repeat(8, auto); gap: 14px 18px; justify-content: start; }
.fa__par { display: flex; flex-direction: column; align-items: center; gap: 6px; }
</style>
