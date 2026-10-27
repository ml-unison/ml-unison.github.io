<script setup lang="ts">
// Tasas por sexo con el bosque calibrado y t* = 1/6.
import datos from '../datos/u03.json'
const E = (datos as any).equidad
const filas = [['prevalencia', 'incumplió (prevalencia)'], ['seleccion', 'marcado de riesgo'], ['tpr', 'TVP: incumplidos detectados'], ['fpr', 'TFP: cumplidos marcados'], ['vpp', 'VPP: marcados que sí incumplieron']]
</script>

<template>
  <div class="eq">
    <div class="eq__leyenda"><span><i style="background: #24398A" />hombres ({{ E.hombres.n.toLocaleString('es-MX') }})</span><span><i style="background: #C9971C" />mujeres ({{ E.mujeres.n.toLocaleString('es-MX') }})</span></div>
    <div v-for="[k, t] in filas" :key="k" class="eq__fila">
      <span class="eq__t">{{ t }}</span>
      <div class="eq__barras">
        <div><div class="eq__b" :style="{ width: `${E.hombres[k] * 100}%`, background: '#24398A' }" /><b>{{ (E.hombres[k] * 100).toFixed(1) }} %</b></div>
        <div><div class="eq__b" :style="{ width: `${E.mujeres[k] * 100}%`, background: '#C9971C' }" /><b>{{ (E.mujeres[k] * 100).toFixed(1) }} %</b></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.eq { display: flex; flex-direction: column; gap: 12px; }
.eq__leyenda { display: flex; gap: 24px; font-size: 18px; } .eq__leyenda i { display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.eq__fila { display: grid; grid-template-columns: 290px 1fr; gap: 16px; align-items: center; }
.eq__t { font-size: 18px; color: var(--tinta); }
.eq__barras > div { display: flex; align-items: center; gap: 8px; height: 22px; }
.eq__b { height: 16px; border-radius: 0 3px 3px 0; max-width: 75%; }
.eq__barras b { font-size: 16px; color: var(--tinta); font-weight: 600; }
</style>
