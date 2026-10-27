<script setup lang="ts">
// Fashion-MNIST: dos ejemplos de cada una de las 10 clases.
import datos from '../datos/u06-03.json'
const D = datos as any
withDefaults(defineProps<{ tam?: number }>(), { tam: 76 })
</script>

<template>
  <div class="pm">
    <div v-for="(ejs, c) in D.mosaico" :key="c" class="pm__clase">
      <div class="pm__imgs"><MapaPixeles v-for="(im, k) in ejs" :key="k" :datos="im" :tam="tam" :vmax="255" :marco="false" /></div>
      <div class="pm__nombre">{{ D.clases[c] }}</div>
    </div>
  </div>
</template>

<style scoped>
.pm { display: grid; grid-template-columns: repeat(5, auto); gap: 16px 22px; justify-content: start; }
.pm__clase { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.pm__imgs { display: flex; gap: 4px; }
.pm__nombre { font-size: 18px; font-weight: 600; color: var(--marino); }
</style>
