<script setup lang="ts">
// EuroSAT (Sentinel-2, RGB, 64 × 64 pixeles = 640 m de lado): dos ejemplos de cada clase de uso de suelo.
import datos from '../datos/u06-03.json'
const E = (datos as any).eurosat
withDefaults(defineProps<{ tam?: number }>(), { tam: 84 })
</script>

<template>
  <div class="ms">
    <div v-for="(ims, c) in E.imagenes" :key="c" class="ms__clase">
      <div class="ms__imgs"><img v-for="(src, k) in ims" :key="k" :src="src" :width="tam" :height="tam" alt="" /></div>
      <div class="ms__nombre">{{ E.clases[c] }}</div>
    </div>
  </div>
</template>

<style scoped>
.ms { display: grid; grid-template-columns: repeat(5, auto); gap: 14px 20px; justify-content: start; }
.ms__clase { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.ms__imgs { display: flex; gap: 4px; }
.ms__imgs img { display: block; border-radius: 3px; }
.ms__nombre { font-size: 17px; font-weight: 600; color: var(--marino); text-align: center; line-height: 1.15; }
</style>
