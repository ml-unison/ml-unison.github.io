<script setup lang="ts">
// (máx − mín) / mín de las distancias de 500 puntos uniformes a un punto, según la dimensión.
import datos from '../datos/u02.json'
const D = (datos as any).dimension
const max = Math.max(...D.razon)
</script>

<template>
  <div class="dd">
    <div v-for="(d, i) in D.d" :key="d" class="dd__fila">
      <span class="dd__d">d = {{ d.toLocaleString('es-MX') }}</span>
      <div class="dd__barra"><div :style="{ width: `${Math.max(0.6, D.razon[i] / max * 100)}%` }" /></div>
      <b>{{ D.razon[i] >= 10 ? D.razon[i].toFixed(0) : D.razon[i].toFixed(2) }}</b>
    </div>
  </div>
</template>

<style scoped>
.dd { display: flex; flex-direction: column; gap: 18px; }
.dd__fila { display: grid; grid-template-columns: 120px 1fr 70px; gap: 16px; align-items: center; font-size: 22px; }
.dd__d { color: var(--tenue); text-align: right; }
.dd__barra { height: 26px; } .dd__barra div { height: 100%; background: var(--azul); border-radius: 0 4px 4px 0; }
.dd b { color: var(--tinta); }
</style>
