<script setup lang="ts">
// Supresión de no máximos en una foto: las cajas crudas de YOLO26n (confianza ≥ 0.25, sin NMS) y las que
// sobreviven a la NMS con IoU 0.7. Cada grupo de duplicados se pinta del color de la caja que lo absorbió.
import { ref, computed } from 'vue'
import datos from '../datos/u06-04.json'
const D = (datos as any).deteccion
const N = D.nms
const F = D.fotos.find((f: any) => f.nombre === N.foto) ?? D.fotos[0]
const despues = ref(false)
const PAL = ['#24398A', '#D5433C', '#C9971C', '#20305A', '#496EA5', '#A87C12', '#B23028', '#8DA3C9', '#6B7385', '#E1B73D', '#5E6678', '#1A2133']
const conservadas = N.crudas.map((c: any, i: number) => (c.conservada ? i : -1)).filter((i: number) => i >= 0)
const color = (c: any, i: number) => PAL[conservadas.indexOf(c.conservada ? i : c.grupo) % PAL.length]
const visibles = computed(() => N.crudas.map((c: any, i: number) => ({ ...c, i })).filter((c: any) => !despues.value || c.conservada))
</script>

<template>
  <div>
    <div class="deslizador" style="gap: 12px">
      <button class="nms-b" :class="{ on: !despues }" @click="despues = false">antes: {{ N.total_crudas }} cajas</button>
      <button class="nms-b" :class="{ on: despues }" @click="despues = true">después: {{ N.total_despues }}</button>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${F.ancho} ${F.alto}`" style="border-radius: 8px">
      <image :href="F.imagen" x="0" y="0" :width="F.ancho" :height="F.alto" />
      <rect v-for="c in visibles" :key="c.i" :x="c.caja[0]" :y="c.caja[1]" :width="c.caja[2] - c.caja[0]" :height="c.caja[3] - c.caja[1]"
            fill="none" :stroke="color(c, c.i)" :stroke-width="c.conservada ? 3.5 : 1.6" :stroke-opacity="c.conservada ? 1 : 0.8" />
    </svg>
  </div>
</template>

<style scoped>
.nms-b { font: inherit; font-size: 19px; padding: 4px 14px; border-radius: 6px; border: 1.5px solid var(--azul); background: #fff; color: var(--azul); cursor: pointer; }
.nms-b.on { background: var(--azul); color: #fff; }
</style>
