<script setup lang="ts">
// Dos nubes en 10 dimensiones, una 8 veces más dispersa que la otra: t-SNE las dibuja del mismo tamaño.
import datos from '../datos/u02.json'
const T = (datos as any).digitos.trampa
const g = T.orig.map((_: any, i: number) => i < 200 ? 0 : 1)
</script>

<template>
  <div class="tt">
    <Dispersion :x="T.orig.map((p: number[]) => p[0])" :y="T.orig.map((p: number[]) => p[1])" :g="g" :ancho="520" :alto="380" :radio="3.5" titulo="Dos primeras coordenadas originales" />
    <Dispersion :x="T.tsne.map((p: number[]) => p[0])" :y="T.tsne.map((p: number[]) => p[1])" :g="g" :ancho="520" :alto="380" :radio="3.5" titulo="t-SNE" />
  </div>
</template>

<style scoped>.tt { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; }</style>
