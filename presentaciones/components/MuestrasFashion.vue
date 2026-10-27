<script setup lang="ts">
// Fashion-MNIST: una columna por clase, con `filas` ejemplos (28×28) de cada una.
import datos from '../datos/u06-02.json'
const D = datos as any
const props = withDefaults(defineProps<{ filas?: number, tam?: number }>(), { filas: 4, tam: 76 })
const clases: string[] = D.clases
const mosaico: string[][] = D.mosaico.map((c: string[]) => c.slice(0, props.filas))
</script>

<template>
  <div class="mf" :style="{ gridTemplateColumns: `repeat(10, ${tam}px)` }">
    <div v-for="(c, k) in clases" :key="k" class="mf__col">
      <img v-for="(src, i) in mosaico[k]" :key="i" :src="src" :width="tam" :height="tam" alt="" />
      <div class="mf__nombre">{{ c }}</div>
    </div>
  </div>
</template>

<style scoped>
.mf { display: grid; gap: 12px; justify-content: center; }
.mf__col { display: flex; flex-direction: column; gap: 6px; align-items: center; }
.mf__col img { image-rendering: pixelated; background: #F6F7FA; border-radius: 6px; display: block; }
.mf__nombre { font-size: 17px; font-weight: 600; color: var(--marino); margin-top: 4px; }
</style>
