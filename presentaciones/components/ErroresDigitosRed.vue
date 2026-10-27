<script setup lang="ts">
// Dígitos de prueba que la red de PyTorch (64-64-10) clasifica mal: real → predicho. Datos: herramientas/laminas/u06_01.py.
import datos from '../datos/u06-01.json'
const E = (datos as any).digitos.errores
</script>

<template>
  <div class="edr">
    <div v-for="(e, i) in E" :key="i" class="edr__d">
      <Digito :px="e.px" :tam="62" marco />
      <div><b>{{ e.real }}</b> → <span style="color: var(--rojo); font-weight: 650">{{ e.pred }}</span></div>
    </div>
  </div>
</template>

<style scoped>
.edr { display: grid; grid-template-columns: repeat(6, auto); gap: 10px 14px; justify-content: start; }
.edr__d { display: flex; flex-direction: column; align-items: center; font-size: 19px; color: var(--tenue); }
</style>
