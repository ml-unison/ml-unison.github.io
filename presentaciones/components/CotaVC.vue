<script setup lang="ts">
// Holgura de la cota VC, Ω(M) = sqrt(8/M · ln(4((2M)^d + 1)/δ)), con d_VC = 3 y δ = 0.05.
import datos from '../datos/u03.json'
const V = (datos as any).cota_vc
</script>

<template>
  <LineasLog :x="V.N" :series="{ 'Ω(M), d_VC = 3': V.omega }" x-log :y-log="false" :y-dominio="[0, 1.3]" :y-ticks="[0, 0.25, 0.5, 0.75, 1, 1.25]"
             :x-ticks="[100, 1000, 10000, 100000, 1000000]" x-titulo="M (escala logarítmica)" y-titulo="E_out − E_in ≤ Ω" :ancho="720" :alto="400" :derecha="180">
    <template #default="{ x, y }">
      <line :x1="x(100)" :x2="x(V.N_para_01)" :y1="y(0.1)" :y2="y(0.1)" stroke="#C9971C" stroke-width="1.5" stroke-dasharray="5 4" />
      <line :x1="x(V.N_para_01)" :x2="x(V.N_para_01)" :y1="y(0)" :y2="y(0.1)" stroke="#C9971C" stroke-width="1.5" stroke-dasharray="5 4" />
      <circle :cx="x(V.N_para_01)" :cy="y(0.1)" r="6" fill="#C9971C" stroke="#fff" stroke-width="2" />
      <text :x="x(V.N_para_01) + 10" :y="y(0.1) - 14" style="fill: #1A2133; font-weight: 600">Ω = 0.1 con M ≈ {{ V.N_para_01.toLocaleString('es-MX') }}</text>
    </template>
  </LineasLog>
</template>
