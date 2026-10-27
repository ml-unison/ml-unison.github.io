<script setup lang="ts">
// Los mismos pingüinos: k-medias (k = 3) contra la mezcla gaussiana que encuentra EM.
import datos from '../datos/u02.json'
const E = (datos as any).em
const fin = E.pasos[E.pasos.length - 1]
// renumerar los grupos de k-medias para que compartan color con la componente más parecida
const mapa: number[] = [0, 1, 2].map(g => { const c = [0, 0, 0]; E.km.forEach((v: number, i: number) => { if (v === g) c[fin.e[i]]++ }); return c.indexOf(Math.max(...c)) })
const km = E.km.map((v: number) => mapa[v])
</script>

<template>
  <div class="kv">
    <div><Dispersion :x="E.x" :y="E.y" :g="km" :ancho="540" :alto="400" :radio="4.6" :dominio="[-2.3, 2.9, -2.3, 2.4]" titulo="k-medias" />
      <div class="kv__ari">ARI contra la especie: <b>{{ E.ari_km }}</b></div></div>
    <div><Dispersion :x="E.x" :y="E.y" :g="fin.e" :ancho="540" :alto="400" :radio="4.6" :dominio="[-2.3, 2.9, -2.3, 2.4]" titulo="Mezcla gaussiana (EM)">
      <template #default="{ x, y }"><polygon v-for="(el, j) in fin.elipses" :key="j" :points="el.map((p: number[]) => `${x(p[0])},${y(p[1])}`).join(' ')" fill="none" :stroke="['#24398A', '#D5433C', '#C9971C'][j]" stroke-width="3" /></template>
    </Dispersion>
      <div class="kv__ari">ARI contra la especie: <b>{{ E.ari_gmm }}</b></div></div>
  </div>
</template>

<style scoped>
.kv { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
.kv__ari { font-size: 21px; color: var(--tenue); margin-left: 14px; } .kv__ari b { color: var(--tinta); }
</style>
