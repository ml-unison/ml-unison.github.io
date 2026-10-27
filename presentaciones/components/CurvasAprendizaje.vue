<script setup lang="ts">
// E_in y E_out esperados contra N, para un modelo simple (grado 1) y uno complejo (grado 5).
import datos from '../datos/u03.json'
const A = (datos as any).aprendizaje
</script>

<template>
  <div class="ca">
    <div v-for="g in ['1', '5']" :key="g">
      <div class="ca__t">grado {{ g }}</div>
      <LineasLog :x="A[g].N" :series="{ 'E_out': A[g].eout, 'E_in': A[g].ein }" :y-log="false" :y-dominio="[0, 0.8]" :y-ticks="[0, 0.2, 0.4, 0.6, 0.8]" :x-dominio="[0, 60]" :x-ticks="[0, 20, 40, 60]"
                 x-titulo="N" :colores="['#D5433C', '#24398A']" :ancho="520" :alto="340" :derecha="90">
        <template #default="{ x, y }">
          <line :x1="x(0)" :x2="x(60)" :y1="y(0.0625)" :y2="y(0.0625)" stroke="#8A91A0" stroke-width="1.5" stroke-dasharray="5 4" />
          
        </template>
      </LineasLog>
    </div>
  </div>
</template>

<style scoped>
.ca { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
.ca__t { font-weight: 600; font-size: 21px; color: var(--marino); }
</style>
