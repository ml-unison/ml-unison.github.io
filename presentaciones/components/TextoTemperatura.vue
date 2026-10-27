<script setup lang="ts">
// Texto generado por el transformer del Quijote a partir del mismo inicio, con distintas temperaturas.
import { ref } from 'vue'
import datos from '../datos/u06-04.json'
const G = (datos as any).generados
const props = withDefaults(defineProps<{ inicial?: string, largo?: number }>(), { inicial: '0.8', largo: 260 })
const tau = ref(props.inicial)
const temps = Object.keys(G.textos).sort((a, b) => Number(a) - Number(b))
</script>

<template>
  <div>
    <div class="tt__botones">
      <span>temperatura τ:</span>
      <button v-for="t in temps" :key="t" :class="{ activo: t === tau }" @click="tau = t">{{ t }}</button>
    </div>
    <div class="tt__texto"><b>{{ G.inicio }}</b>{{ G.textos[tau].slice(G.inicio.length, G.inicio.length + largo) }}…</div>
  </div>
</template>

<style scoped>
.tt__botones { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; font-size: 21px; color: var(--tenue); }
.tt__botones button { font: 600 19px var(--f-texto); padding: 4px 16px; border-radius: 18px; border: 2px solid var(--rojo); background: #fff; color: var(--rojo); cursor: pointer; }
.tt__botones button.activo { background: var(--rojo); color: #fff; }
.tt__texto { white-space: pre-wrap; font-family: var(--f-texto); font-size: 18.5px; line-height: 1.45; background: var(--fondo-alt); border-radius: 8px; padding: 16px 22px; color: var(--tinta); }
.tt__texto b { color: var(--azul); font-weight: 650; }
</style>
