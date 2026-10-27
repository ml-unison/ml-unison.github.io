<script setup lang="ts">
// Rentas promedio por hora de cada mes: 2011 (con lo que se entrenó) contra 2012 (producción).
import d from '../datos/u08-02.json'
const M = (d as any).mensual
const meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const x = Array.from({ length: 12 }, (_, i) => i + 1)
const props = withDefaults(defineProps<{ futuro?: boolean }>(), { futuro: true })
const series = props.futuro ? { '2011 · entrenamiento': M.y2011, '2012 · producción': M.y2012 } : { '2011 · entrenamiento': M.y2011, ' ': M.y2011.map(() => null) }
</script>

<template>
  <LineasLog :x="x" :series="series" :y-log="false" :y-dominio="[0, 320]" :y-ticks="[0, 100, 200, 300]"
             :x-dominio="[1, 12]" :x-ticks="[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]" :formato-x="(v: number) => meses[v - 1]"
             :colores="['#24398A', '#D5433C']" marcadores y-titulo="bicicletas por hora (promedio del mes)" :ancho="760" :alto="400" :derecha="200" />
</template>
