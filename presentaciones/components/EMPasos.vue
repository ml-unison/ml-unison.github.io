<script setup lang="ts">
// EM paso a paso en los pingüinos (pico, estandarizado): elipses de 1 desviación,
// color = componente más probable, opacidad = certeza. A la derecha, la log-verosimilitud.
import { computed, ref } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u02.json'
import { SERIE } from './colores'
const E = (datos as any).em
const k = ref(0)
const P = computed(() => E.pasos[k.value])
const W = 380, H = 240, m = { l: 66, r: 12, t: 40, b: 40 }
const n = E.ll.length
const xl = scaleLinear().domain([0, n - 1]).range([m.l, W - m.r])
const yl = scaleLinear().domain([-1200, -800]).range([H - m.b, m.t])
const camino = computed(() => line<number>().x((_, i) => xl(i)).y(v => yl(Math.max(v, -1200)))(E.ll.slice(0, P.value.it + 1))!)
</script>

<template>
  <div class="em">
    <Dispersion :x="E.x" :y="E.y" :g="P.e" :opacidad="P.resp.map((r: number) => 0.15 + 0.75 * (r - 0.33) / 0.67)" :ancho="620" :alto="450" :radio="4.6" :dominio="[-2.3, 2.9, -2.3, 2.4]">
      <template #default="{ x, y }">
        <polygon v-for="(el, j) in P.elipses" :key="j" :points="el.map((p: number[]) => `${x(p[0])},${y(p[1])}`).join(' ')" fill="none" :stroke="SERIE[j]" stroke-width="3" />
      </template>
    </Dispersion>
    <div class="em__lado">
      <div class="em__botones">
        <button v-for="(p, i) in E.pasos" :key="i" :class="{ activo: i === k }" @click="k = i">{{ p.it }}</button>
      </div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <g class="eje"><line :x1="m.l" :x2="W - m.r" :y1="H - m.b" :y2="H - m.b" />
          <text v-for="t in [0, 50, 100]" :key="t" :x="xl(t)" :y="H - m.b + 24" text-anchor="middle">{{ t }}</text>
          <text v-for="t in [-1200, -1100, -1000, -900, -800]" :key="`y${t}`" :x="m.l - 8" :y="yl(t) + 6" text-anchor="end">{{ String(t).replace('-', '−') }}</text>
        </g>
        <path :d="camino" fill="none" :stroke="SERIE[0]" stroke-width="3" />
        <text :x="m.l - 58" :y="18" class="titulo-eje" style="font-size: 17px">log-verosimilitud por iteración</text>
      </svg>
      <div class="em__ll">iteración {{ P.it }} · log-verosimilitud = <b>{{ E.ll[P.it].toFixed(1).replace('-', '−') }}</b></div>
      <p class="em__nota">Pico de 342 pingüinos, largo (horizontal) y profundidad (vertical), estandarizados. Inicio: tres pingüinos al azar con covarianza identidad.</p>
    </div>
  </div>
</template>

<style scoped>
.em { display: grid; grid-template-columns: 620px 1fr; gap: 36px; align-items: start; }
.em__lado { display: flex; flex-direction: column; gap: 12px; }
.em__botones { display: flex; gap: 8px; flex-wrap: wrap; }
.em__botones button { font: 600 18px var(--f-texto); padding: 4px 12px; border-radius: 6px; border: 1.5px solid var(--azul-claro); background: #fff; color: var(--azul); cursor: pointer; }
.em__botones button.activo { background: var(--azul); color: #fff; border-color: var(--azul); }
.em__ll { font-size: 23px; } .em__ll b { color: var(--azul); }
.em__nota { font-size: 17px; color: var(--tenue); }
</style>
