<script setup lang="ts">
// P[|ν − μ| > ε] exacta (binomial, μ = 0.5) contra la cota de Hoeffding, según M.
import datos from '../datos/u03.json'
const H = (datos as any).hoeffding
withDefaults(defineProps<{ eps?: string }>(), { eps: '0.1' })
</script>

<template>
  <LineasLog :x="H.N" :series="{ 'cota de Hoeffding': H.eps[eps].cota, 'probabilidad exacta': H.eps[eps].exacto }" :y-dominio="[1e-6, 1]"
             :y-ticks="[1e-6, 1e-4, 1e-2, 1]" :formato-y="(v: number) => v === 1 ? '1' : `10${String(Math.round(Math.log10(v))).replace('-', '⁻').replace(/\d/g, (d: string) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d])}`"
             :x-ticks="[0, 100, 200, 300, 400]" x-titulo="tamaño de la muestra M" :y-titulo="`P[|ν − μ| > ${eps}]`" :colores="['#D5433C', '#24398A']" :punteadas="['cota de Hoeffding']" :ancho="760" :alto="420" :derecha="190" />
</template>
