<script setup lang="ts">
// Cuatro tipos de covarianza ajustados a los mismos pingüinos.
import datos from '../datos/u02.json'
const E = (datos as any).em, C = (datos as any).covarianzas
const tipos = [['spherical', 'Esférica', 'σ²ₖ I', 'k'], ['diag', 'Diagonal', 'diag(σ²ₖ₁, …)', 'kd'], ['tied', 'Compartida', 'Σ igual para todas', 'd(d+1)/2'], ['full', 'Completa', 'Σₖ libre', 'k·d(d+1)/2']]
</script>

<template>
  <div class="cv">
    <div v-for="t in tipos" :key="t[0]">
      <Dispersion :x="E.x" :y="E.y" :g="E.x.map(() => 3)" :opacidad="0.35" :ancho="290" :alto="250" :radio="3" :dominio="[-2.3, 2.9, -2.3, 2.4]">
        <template #default="{ x, y }"><polygon v-for="(el, j) in C[t[0]].elipses" :key="j" :points="el.map((p: number[]) => `${x(p[0])},${y(p[1])}`).join(' ')" fill="none" :stroke="['#24398A', '#D5433C', '#C9971C'][j]" stroke-width="2.6" /></template>
      </Dispersion>
      <div class="cv__n">{{ t[1] }}</div>
      <div class="cv__t">{{ t[2] }}</div>
      <div class="cv__t">parámetros de covarianza: {{ t[3] }}</div>
      <div class="cv__t">ARI: <b>{{ C[t[0]].ari }}</b></div>
    </div>
  </div>
</template>

<style scoped>
.cv { display: grid; grid-template-columns: repeat(4, 1fr); gap: 22px; }
.cv__n { font: 650 16px var(--f-titulo); letter-spacing: .08em; text-transform: uppercase; color: var(--azul); margin-top: 4px; }
.cv__t { font-size: 18px; color: var(--tenue); } .cv__t b { color: var(--tinta); }
</style>
