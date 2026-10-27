<script setup lang="ts">
// Perfil por hora en días laborales de prueba: real contra lo que predicen M2, M3 y M4.
import datos from '../datos/u04.json'
const P = (datos as any).perfil_modelos
const h = Array.from({ length: 24 }, (_, i) => i)
</script>

<template>
  <LineasLog :x="h" :series="{ 'observado': P.real, 'M2 (hora como número)': P.M2, 'M3 (hora one-hot)': P.M3, 'M4 (hora × laboral)': P.M4 }" :y-log="false"
             :y-dominio="[-50, 550]" :y-ticks="[0, 100, 200, 300, 400, 500]" :x-dominio="[0, 23]" :x-ticks="[0, 4, 8, 12, 16, 20]"
             :colores="['#1A2133', '#6B7385', '#C9971C', '#D5433C']" :punteadas="['M2 (hora como número)']" x-titulo="hora (días laborales de prueba)" y-titulo="bicicletas por hora (promedio)" :ancho="760" :alto="420" :derecha="250" />
</template>
