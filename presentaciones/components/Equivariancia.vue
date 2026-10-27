<script setup lang="ts">
// Trasladar y luego filtrar = filtrar y luego trasladar (botín con margen de ceros, filtro de Sobel vertical).
import datos from '../datos/u06-03.json'
const E = (datos as any).equivariancia
const lim = Math.max(...E.y.flat().map(Math.abs))
withDefaults(defineProps<{ tam?: number }>(), { tam: 170 })
</script>

<template>
  <div class="eq">
    <div />
    <div class="eq__col">original</div>
    <div class="eq__col">trasladada ({{ E.a }} abajo, {{ E.b }} a la derecha)</div>
    <div class="eq__fila">imagen</div>
    <MapaPixeles :datos="E.x" :tam="tam" :vmax="255" />
    <MapaPixeles :datos="E.xt" :tam="tam" :vmax="255" />
    <div class="eq__fila">mapa del filtro</div>
    <MapaPixeles :datos="E.y" :tam="tam" modo="divergente" :vmax="lim" />
    <MapaPixeles :datos="E.yt" :tam="tam" modo="divergente" :vmax="lim" />
  </div>
</template>

<style scoped>
.eq { display: grid; grid-template-columns: auto auto auto; gap: 10px 22px; align-items: center; justify-items: center; }
.eq__col { font-size: 17px; color: var(--tenue); font-weight: 600; text-align: center; max-width: 190px; line-height: 1.2; }
.eq__fila { font-size: 17px; color: var(--tenue); font-weight: 600; justify-self: end; }
</style>
