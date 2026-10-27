<script setup lang="ts">
defineProps<{ numero?: string | number }>()
</script>

<template>
  <div class="slidev-layout seccion">
    <div v-if="numero" class="seccion__numero">{{ String(numero).padStart(2, '0') }}</div>
    <div class="seccion__regla" />
    <slot />
    <div class="seccion__fondo"><MotivoFrontera :ancho="620" :alto="460" :n="70" :semilla="Number(numero ?? 3) * 11" /></div>
  </div>
</template>
