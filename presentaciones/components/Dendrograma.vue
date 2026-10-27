<script setup lang="ts">
// Dendrograma (enlace promedio) de las 10 presas según su perfil de llenado mensual,
// horizontal: hojas a la izquierda con nombre, río y una miniserie. El deslizador corta
// el árbol; el punto de color junto a cada nombre indica su grupo.
import { computed, ref } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u02.json'
import { SERIE } from './colores'
const P = (datos as any).presas
const W = 1080, H = 440, izq = 440, m = { r: 30, t: 36, b: 62 }
const maxd = Math.max(...P.dcoord.flat())
const x = scaleLinear().domain([0, maxd]).range([izq, W - m.r])
const y = scaleLinear().domain([0, P.hojas.length * 10]).range([m.t, H - m.b])
const corte = ref(Math.round(maxd * 0.62))
const sx = scaleLinear().domain([0, P.meses.length - 1]).range([190, 380])
const sy = (v: number, j: number) => y(j * 10 + 5) + 16 - v / 100 * 34
const mini = (n: string, j: number) => line<number>().x((_, i) => sx(i)).y(v => sy(v, j))(P.series[n])!
const ramas = computed(() => P.icoord.map((ic: number[], k: number) => ic.map((v: number, i: number) => `${x(P.dcoord[k][i])},${y(v)}`).join(' ')))
// grupos al cortar: unión de los enlaces por debajo del corte (matriz de enlace de scipy)
const grupos = computed(() => {
  const n = P.orden.length, padre = Array.from({ length: 2 * n }, (_, i) => i)
  const raiz = (i: number): number => padre[i] === i ? i : (padre[i] = raiz(padre[i]))
  P.Z.forEach((z: number[], k: number) => { if (z[2] <= corte.value) { padre[raiz(z[0])] = n + k; padre[raiz(z[1])] = n + k } })
  const ids: number[] = [], g: number[] = []
  for (const hoja of P.orden) { const rz = raiz(hoja); if (!ids.includes(rz)) ids.push(rz); g.push(ids.indexOf(rz)) }
  return { g, k: ids.length }
})
const color = (j: number) => grupos.value.k > 4 ? '#B9BEC9' : SERIE[grupos.value.g[j]]
</script>

<template>
  <div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <text :x="sx(0)" :y="18" style="font-size: 15px">llenado ene 2020 – jun 2023 (% del NAMO)</text>
      <g v-for="(h, j) in P.hojas" :key="h">
        <circle :cx="8" :cy="y(j * 10 + 5)" r="7" :fill="color(j)" />
        <text :x="24" :y="y(j * 10 + 5) + 6" style="fill: #1A2133; font-weight: 600">{{ h }}</text>
        <text :x="24" :y="y(j * 10 + 5) + 24" style="font-size: 14px">río {{ P.rio[j] === 'Otros' ? '— (otro)' : P.rio[j] }}</text>
        <line :x1="sx(0)" :x2="sx(P.meses.length - 1)" :y1="sy(0, j)" :y2="sy(0, j)" stroke="#E2E5EC" />
        <path :d="mini(h, j)" fill="none" stroke="#24398A" stroke-width="2" />
      </g>
      <polyline v-for="(p, k) in ramas" :key="k" :points="p" fill="none" stroke="#20305A" stroke-width="2.2" />
      <line :x1="x(corte)" :x2="x(corte)" :y1="m.t - 10" :y2="H - m.b + 4" stroke="#D5433C" stroke-width="2" stroke-dasharray="6 5" />
      <g class="eje"><line :x1="izq" :x2="W - m.r" :y1="H - m.b + 10" :y2="H - m.b + 10" />
        <text v-for="t in x.ticks(5)" :key="t" :x="x(t)" :y="H - m.b + 34" text-anchor="middle">{{ t }}</text>
        <text class="titulo-eje" :x="(izq + W - m.r) / 2" :y="H - 4" text-anchor="middle" style="font-size: 16px">distancia a la que se unen (puntos porcentuales)</text>
      </g>
    </svg>
    <div class="deslizador" style="margin-top: 4px"><span>Cortar el árbol en</span><input v-model.number="corte" type="range" min="0" :max="Math.ceil(maxd)" step="1" /><b>{{ corte }}</b><span class="tenue">→ {{ grupos.k }} grupos</span></div>
  </div>
</template>
