<script setup lang="ts">
// RECM de cada mes de 2012 con cuatro políticas de reentrenamiento; la línea punteada gris es el error de validación.
import d from '../datos/u08-02.json'
const D = d as any
const props = withDefaults(defineProps<{ series?: number }>(), { series: 4 })
const meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
const x = Array.from({ length: 12 }, (_, i) => i + 1)
const P = D.politicas.slice(0, props.series) as { nombre: string, recm: number[] }[]
const series = Object.fromEntries(P.map(p => [p.nombre, p.recm]))
const colores = ['#6B7385', '#C9971C', '#24398A', '#D5433C']
</script>

<template>
  <LineasLog :x="x" :series="series" :y-log="false" :y-dominio="[0, 175]" :y-ticks="[0, 50, 100, 150]"
             :x-dominio="[1, 12]" :x-ticks="x" :formato-x="(v: number) => meses[v - 1]" :colores="colores"
             :punteadas="['3 meses, etiquetas 3 meses tarde']" marcadores y-titulo="RECM del mes (bicicletas por hora)" :ancho="1060" :alto="380" :derecha="330">
    <template #default="{ x: sx, y: sy }">
      <line :x1="sx(1)" :x2="sx(12)" :y1="sy(D.validacion.recm)" :y2="sy(D.validacion.recm)" stroke="#5E6678" stroke-width="1.6" stroke-dasharray="3 5" />
      <text :x="sx(12)" :y="sy(D.validacion.recm) + 24" text-anchor="end" style="font-size: 16px">validación: {{ D.validacion.recm }}</text>
    </template>
  </LineasLog>
</template>
