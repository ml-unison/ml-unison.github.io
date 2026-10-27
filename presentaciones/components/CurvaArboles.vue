<script setup lang="ts">
// Error de clasificación (censo) contra número de árboles: bagging y bosque, en prueba y fuera de la bolsa.
import datos from '../datos/u05-02.json'
const C = (datos as any).curvas
const b = C.error_bag.map((_: number, i: number) => i + 1)
const series = {
  'bagging · prueba': C.error_bag, 'bagging · OOB': C.oob_error_bag,
  'bosque · prueba': C.error_rf, 'bosque · OOB': C.oob_error_rf,
}
</script>

<template>
  <LineasLog :x="b" :series="series" :y-log="false" :y-dominio="[0.14, 0.2]" :y-ticks="[0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]" :x-dominio="[1, 300]" :x-ticks="[1, 50, 100, 150, 200, 250, 300]"
             :colores="['#6B7385', '#6B7385', '#24398A', '#24398A']" :punteadas="['bagging · OOB', 'bosque · OOB']"
             x-titulo="número de árboles, B" y-titulo="error de clasificación" :ancho="780" :alto="430" :derecha="190">
    <template #default="{ x, y }">
      <line :x1="x(1)" :x2="x(300)" :y1="y(C.error_arbol)" :y2="y(C.error_arbol)" stroke="#D5433C" stroke-width="2" stroke-dasharray="3 4" />
      <text :x="x(300)" :y="y(C.error_arbol) - 10" text-anchor="end" style="fill: #1A2133; font-weight: 600">un árbol profundo solo: {{ C.error_arbol.toFixed(3) }}</text>
    </template>
  </LineasLog>
</template>
