<script setup lang="ts">
// Largo contra profundidad del pico de 342 pingüinos; con `especies`, coloreados y etiquetados.
import datos from '../datos/u02.json'
const P = (datos as any).pinguinos
withDefaults(defineProps<{ especies?: boolean, ancho?: number, alto?: number }>(), { especies: false, ancho: 720, alto: 470 })
const med = (k: number, v: number[]) => { const s = v.filter((_, i) => P.especie[i] === k).sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }
const etiquetas = [0, 1, 2].map(k => ({ x: med(k, P.largo) + [-4.2, 4.6, 1.5][k], y: med(k, P.prof) + [2.4, 1.6, -2.3][k], texto: P.nombres[k] }))
</script>

<template>
  <Dispersion :x="P.largo" :y="P.prof" :g="especies ? P.especie : P.largo.map(() => 3)" :etiquetas="especies ? etiquetas : []"
              :ancho="ancho" :alto="alto" ejes x-titulo="largo del pico (mm)" y-titulo="profundidad del pico (mm)" :radio="5" :opacidad="0.75" :dominio="[31, 61, 12.5, 22]" />
</template>
