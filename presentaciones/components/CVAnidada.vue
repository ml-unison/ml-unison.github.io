<script setup lang="ts">
// CV anidada: el ciclo externo estima el error; el interno, dentro de cada entrenamiento, elige λ.
const W = 760, x0 = 120, ancho = 560, k = 5
const seg = ancho / k
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} 330`" role="img">
    <text x="0" y="40" style="font-size: 17px; font-weight: 600; fill: #1A2133">externo</text>
    <rect v-for="j in k" :key="j" :x="x0 + (j - 1) * seg" y="20" :width="seg - 6" height="30" rx="4" :fill="j === 2 ? '#D5433C' : '#C0CEE3'" />
    <text :x="x0 + seg + seg / 2 - 3" y="72" text-anchor="middle" style="font-size: 15px; fill: #1A2133">prueba externa</text>
    <path :d="`M${x0},58 L${x0 + 40},120 M${x0 + ancho - 6},58 L${x0 + ancho - 46},120`" stroke="#8A91A0" stroke-dasharray="4 4" fill="none" />
    <g v-for="i in 4" :key="i" :transform="`translate(0, ${110 + (i - 1) * 36})`">
      <text x="0" y="22" style="font-size: 16px">interno {{ i }}</text>
      <rect v-for="j in 4" :key="j" :x="x0 + 40 + (j - 1) * ((ancho - 80) / 4)" y="4" :width="(ancho - 80) / 4 - 5" height="24" rx="4" :fill="i === j ? '#C9971C' : '#C0CEE3'" />
    </g>
    <text :x="x0 + ancho / 2" y="270" text-anchor="middle" style="font-size: 17px; fill: #1A2133">con las 4 partes de entrenamiento: CV interna para elegir λ, reentrenar con λ elegido</text>
    <text :x="x0 + ancho / 2" y="300" text-anchor="middle" style="font-size: 17px; fill: #1A2133">y evaluar en la prueba externa. Repetir para las 5 particiones externas.</text>
  </svg>
</template>
