<script setup lang="ts">
// Las cuatro medidas de los pingüinos proyectadas a 2 dimensiones: PCA (sin etiquetas) y LDA de Fisher (con etiquetas).
import datos from '../datos/u04.json'
const F = (datos as any).fisher
</script>

<template>
  <div class="fp">
    <div><Dispersion :x="F.pca[0]" :y="F.pca[1]" :g="F.g" :ancho="520" :alto="380" titulo="PCA: máxima varianza" :radio="4.5" /></div>
    <div><Dispersion :x="F.lda[0]" :y="F.lda[1]" :g="F.g" :ancho="520" :alto="380" titulo="LDA: máxima separación entre especies" :radio="4.5" /></div>
  </div>
</template>

<style scoped>
.fp { display: grid; grid-template-columns: 1fr 1fr; gap: 36px; }
.fp > div { background: #F6F7FA; border-radius: 8px; }
</style>
