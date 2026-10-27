<script setup lang="ts">
// Un mismo dígito reconstruido con m componentes.
import datos from '../datos/u02.json'
const G = (datos as any).digitos
const ms = Object.keys(G.recon).map(Number).sort((a, b) => a - b)
</script>

<template>
  <div class="rc">
    <div v-for="m in ms" :key="m" class="rc__item"><Imagen8 :px="G.recon[m]" :tam="130" /><span>m = {{ m }}</span></div>
    <div class="rc__item rc__orig"><Imagen8 :px="G.original" :tam="130" /><span>original (64)</span></div>
  </div>
</template>

<style scoped>
.rc { display: flex; gap: 22px; align-items: flex-end; }
.rc__item { display: flex; flex-direction: column; align-items: center; gap: 6px; font-size: 19px; color: var(--tenue); }
.rc__orig { margin-left: 18px; padding-left: 26px; border-left: 1px solid var(--linea); }
</style>
