<script setup lang="ts">
// Raíces del polinomio característico en el plano complejo: estacionario si todas quedan fuera del círculo unitario.
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u07-02.json'
const A = (datos as any).ar2
const props = withDefaults(defineProps<{ conAcf?: boolean }>(), { conAcf: false })
const S = 400, m = 34
const x = scaleLinear().domain([-1.7, 1.7]).range([m, S - m])
const y = scaleLinear().domain([-1.7, 1.7]).range([S - m, m])
const r = x(1) - x(0)
const puntos = [
  { re: A.raiz[0], im: A.raiz[1], c: '#24398A' },
  { re: A.raiz[0], im: -A.raiz[1], c: '#24398A' },
  { re: 1 / 0.9, im: 0, c: '#C9971C' },
  { re: 1, im: 0, c: '#D5433C' },
]
</script>

<template>
  <div :class="{ ra2: conAcf }">
  <svg class="grafica" :viewBox="`0 0 ${S + 190} ${S}`" role="img">
    <line :x1="x(-1.7)" :x2="x(1.7)" :y1="y(0)" :y2="y(0)" stroke="#8A91A0" stroke-width="1.2" />
    <line :x1="x(0)" :x2="x(0)" :y1="y(-1.7)" :y2="y(1.7)" stroke="#8A91A0" stroke-width="1.2" />
    <circle :cx="x(0)" :cy="y(0)" :r="r" fill="#FCEFEE" stroke="#D5433C" stroke-width="2" stroke-dasharray="6 5" />
    <text :x="x(0) + 8" :y="y(0.55)" style="fill: #B23028; font-weight: 600; font-size: 17px">no estacionario</text>
    <text :x="x(1.62)" :y="y(0) + 24" text-anchor="end" style="font-size: 16px">Re</text>
    <text :x="x(0) + 8" :y="y(1.6)" style="font-size: 16px">Im</text>
    <text :x="x(1)" :y="y(0) + 22" text-anchor="middle" style="font-size: 15px">1</text>
    <circle v-for="(p, i) in puntos" :key="i" :cx="x(p.re)" :cy="y(p.im)" r="8" :fill="p.c" stroke="#fff" stroke-width="2" />
    <text :x="x(A.raiz[0]) + 14" :y="y(A.raiz[1]) + 6" style="fill: #1A2133; font-weight: 600">0.75 + 1.20i</text>
    <text :x="x(A.raiz[0]) + 14" :y="y(-A.raiz[1]) + 6" style="fill: #1A2133; font-weight: 600">0.75 − 1.20i</text>
    <text :x="S - 10" :y="y(0) - 40" style="fill: #1A2133; font-weight: 600">1/0.9 = 1.11</text>
    <text :x="S - 10" :y="y(0) - 16" style="fill: #5E6678; font-size: 16px">AR(1), φ = 0.9</text>
    <text :x="S - 10" :y="y(0) + 40" style="fill: #B23028; font-weight: 600">z = 1</text>
    <text :x="S - 10" :y="y(0) + 62" style="fill: #5E6678; font-size: 16px">caminata aleatoria</text>
  </svg>
  <CorrelogramaARIMA v-if="conAcf" :valores="A.acf" :banda="Math.round(1960 / Math.sqrt(1000)) / 1000" titulo="ACF del AR(2) simulado: oscila y se apaga" :ancho="520" :alto="300" />
  </div>
</template>

<style scoped>
.ra2 { display: grid; grid-template-columns: 470px 560px; gap: 40px; align-items: center; }
</style>
