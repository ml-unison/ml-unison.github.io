<script setup lang="ts">
// Renta media de bicicletas por hora del día, días laborales contra fines de semana y feriados.
import datos from '../datos/u04.json'
const P = (datos as any).perfil
</script>

<template>
  <LineasLog :x="P.hora" :series="{ 'día laboral': P.laboral, 'fin de semana o feriado': P.descanso }" :y-log="false" :y-dominio="[0, 500]" :y-ticks="[0, 100, 200, 300, 400, 500]"
             :x-dominio="[0, 23]" :x-ticks="[0, 4, 8, 12, 16, 20]" x-titulo="hora del día" y-titulo="bicicletas rentadas por hora (promedio)" :ancho="780" :alto="420" :derecha="230" marcadores />
</template>
