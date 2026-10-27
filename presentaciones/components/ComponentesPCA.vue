<script setup lang="ts">
// El dígito promedio y las primeras 8 componentes principales, como imágenes.
import datos from '../datos/u02.json'
const G = (datos as any).digitos
</script>

<template>
  <div class="cp">
    <div class="cp__item"><Imagen8 :px="G.media" :tam="100" /><span>promedio</span></div>
    <div v-for="(c, i) in G.comps" :key="i" class="cp__item"><Imagen8 :px="c" :tam="100" modo="divergente" /><span>PC{{ i + 1 }}</span></div>
  </div>
</template>

<style scoped>
.cp { display: grid; grid-template-columns: repeat(9, 1fr); gap: 14px; }
.cp__item { display: flex; flex-direction: column; align-items: center; gap: 6px; font-size: 18px; color: var(--tenue); }
.cp__item:first-child { padding-right: 14px; border-right: 1px solid var(--linea); }
</style>
