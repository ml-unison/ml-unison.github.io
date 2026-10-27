<script setup lang="ts">
// Campo receptivo de una neurona en cada etapa de la red: 3, 4, 8 y 10 pixeles de lado.
// Los cuadros corresponden a la posición (3, 3) del mapa final de 7 × 7 (pixeles 9–18).
import { ref } from 'vue'
import datos from '../datos/u06-03.json'
const D = datos as any
const etapas = [
  { nombre: 'conv1', r: D.campo.teorico[0], x: 11, color: '#C9971C' },
  { nombre: '+ max-pool', r: D.campo.teorico[1], x: 11, color: '#6B7385' },
  { nombre: 'conv2', r: D.campo.teorico[2], x: 9, color: '#D5433C' },
  { nombre: '+ max-pool', r: D.campo.teorico[3], x: 9, color: '#24398A' },
]
const sel = ref(3)
withDefaults(defineProps<{ tam?: number }>(), { tam: 380 })
</script>

<template>
  <div class="cr">
    <MapaPixeles :datos="D.sobel.prenda" :tam="tam" :vmax="255" rejilla>
      <rect v-for="(e, k) in etapas" v-show="k <= sel" :key="k" :x="e.x" :y="e.x" :width="e.r" :height="e.r"
            fill="none" :stroke="e.color" :stroke-width="k === sel ? 0.28 : 0.16" :stroke-dasharray="k === sel ? 'none' : '0.4 0.25'" />
    </MapaPixeles>
    <div class="cr__lista">
      <button v-for="(e, k) in etapas" :key="k" :class="{ activo: k === sel }" @click="sel = k">
        <span class="cr__muestra" :style="{ borderColor: e.color }" />
        <span>{{ e.nombre }}</span><b>{{ e.r }} × {{ e.r }}</b>
      </button>
    </div>
  </div>
</template>

<style scoped>
.cr { display: flex; gap: 30px; align-items: center; }
.cr__lista { display: flex; flex-direction: column; gap: 10px; }
.cr__lista button { font: inherit; font-size: 20px; display: grid; grid-template-columns: 26px 1fr auto; gap: 12px; align-items: center; min-width: 260px; text-align: left; padding: 6px 12px; border: 1.5px solid var(--linea); border-radius: 8px; background: #fff; color: var(--tinta); cursor: pointer; }
.cr__lista button.activo { border-color: var(--azul); background: var(--tinte-azul); }
.cr__lista b { color: var(--marino); font-variant-numeric: tabular-nums; }
.cr__muestra { width: 22px; height: 22px; border: 3px solid; border-radius: 3px; }
</style>
