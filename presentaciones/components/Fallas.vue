<script setup lang="ts">
// Tres conjuntos donde k-medias se equivoca; con metodo="db", lo que hace DBSCAN.
import datos from '../datos/u02.json'
const F = (datos as any).fallas
const props = withDefaults(defineProps<{ metodo?: 'km' | 'db' }>(), { metodo: 'km' })
const titulos = ['Escalas distintas por eje', 'Formas no convexas', 'Tamaños y densidades distintos']
</script>

<template>
  <div class="fa">
    <div v-for="(f, i) in F" :key="i">
      <Dispersion :x="f.x" :y="f.y" :g="f[metodo]" :ancho="370" :alto="300" :radio="4" />
      <div class="fa__t">{{ titulos[i] }}</div>
    </div>
  </div>
</template>

<style scoped>
.fa { display: grid; grid-template-columns: repeat(3, 1fr); gap: 30px; }
.fa__t { text-align: center; font-size: 20px; color: var(--tinta); font-weight: 600; margin-top: 6px; }
</style>
