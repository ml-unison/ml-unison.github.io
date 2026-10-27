<script setup lang="ts">
import datos from '../datos/u01.json'

const props = withDefaults(defineProps<{ conjunto?: 'mosaico' | 'sietes' | 'unos', tam?: number, columnas?: number, color?: string }>(),
  { conjunto: 'mosaico', tam: 64, columnas: 10, color: '#20305A' })
const D = datos.digitos as any
const lista: number[][] = props.conjunto === 'mosaico' ? D.mosaico.flat() : D[props.conjunto]
</script>

<template>
  <div class="mosaico" :style="{ gridTemplateColumns: `repeat(${columnas}, ${tam}px)` }">
    <Digito v-for="(d, i) in lista" :key="i" :px="d" :tam="tam" :color="color" marco />
  </div>
</template>

<style scoped>
.mosaico { display: grid; gap: 14px; justify-content: center; }
</style>
