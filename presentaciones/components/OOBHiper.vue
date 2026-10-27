<script setup lang="ts">
// AUC fuera de la bolsa del bosque (censo) contra m (atributos por división) y contra el tamaño mínimo de hoja.
import datos from '../datos/u05-02.json'
const H = (datos as any).hiper
</script>

<template>
  <div class="oh">
    <div>
      <div class="oh__t">Atributos por división, m <span>(hojas de un dato)</span></div>
      <LineasLog :x="H.m" :series="{ ' ': H.auc_m }" :y-log="false" :y-dominio="[0.89, 0.92]" :y-ticks="[0.89, 0.9, 0.91, 0.92]" :x-dominio="[1, 12]" :x-ticks="H.m"
                 :formato-y="(v: number) => v.toFixed(2)" marcadores x-titulo="m   (m = 12: bagging)" y-titulo="AUC fuera de la bolsa" :ancho="540" :alto="360" :derecha="20" />
    </div>
    <div>
      <div class="oh__t">Tamaño mínimo de hoja <span>(m = {{ H.m_mejor }})</span></div>
      <LineasLog :x="H.hojas" :series="{ ' ': H.auc_hoja }" x-log :y-log="false" :y-dominio="[0.89, 0.92]" :y-ticks="[0.89, 0.9, 0.91, 0.92]" :x-dominio="[1, 50]" :x-ticks="H.hojas"
                 :formato-y="(v: number) => v.toFixed(2)" marcadores x-titulo="datos por hoja (como mínimo)" y-titulo="AUC fuera de la bolsa" :ancho="540" :alto="360" :derecha="20" />
    </div>
  </div>
</template>

<style scoped>
.oh { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
.oh__t { font-weight: 650; color: var(--marino); font-size: 21px; margin-bottom: 4px; }
.oh__t span { font-weight: 400; color: var(--tenue); font-size: 18px; }
</style>
