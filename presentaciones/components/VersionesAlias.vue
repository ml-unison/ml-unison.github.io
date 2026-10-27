<script setup lang="ts">
// Línea de tiempo del modelo registrado «riesgo-credito»: las versiones se quedan, los alias se mueven.
// paso 0: v1 es champion · paso 1: llega v2 como challenger · paso 2: champion pasa a v2.
import { computed } from 'vue'
import datos from '../datos/u08-01.json'
const D = (datos as any).duelo
const props = withDefaults(defineProps<{ paso?: number }>(), { paso: 2 })
const W = 1120, H = 400, yl = 270
const xv1 = 270, xv2 = 700, xv3 = 1010
const p = computed(() => Math.min(props.paso, 2))
const xCampeon = computed(() => (p.value >= 2 ? xv2 : xv1))
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <!-- quién consulta el registro -->
    <text :x="W / 2" y="30" text-anchor="middle" style="fill: #1A2133; font-size: 20px">
      el sistema en producción siempre pide <tspan style="font-family: var(--f-codigo); fill: #20305A; font-weight: 600">models:/riesgo-credito@champion</tspan>
    </text>
    <line :x1="W / 2" y1="44" :x2="xCampeon" y2="98" stroke="#C9971C" stroke-width="2" stroke-dasharray="6 5" class="mov" />

    <!-- tiempo -->
    <line x1="60" :x2="xv2 + 150" :y1="yl" :y2="yl" stroke="#8A91A0" stroke-width="3" />
    <line :x1="xv2 + 150" :x2="W - 30" :y1="yl" :y2="yl" stroke="#C5C9D3" stroke-width="3" stroke-dasharray="8 8" />
    <text x="60" :y="yl - 14" style="fill: #5E6678; font-size: 17px">tiempo →</text>

    <!-- versiones -->
    <g v-for="v in [{ x: xv1, n: 1, fam: 'regresión logística', cuando: 'hace seis meses', auc: D.v1.auc, ver: true },
                    { x: xv2, n: 2, fam: 'bosque aleatorio', cuando: 'hoy', auc: D.v2.auc, ver: p >= 1 }]" :key="v.n">
      <g :style="{ opacity: v.ver ? 1 : 0, transition: 'opacity .5s' }">
        <circle :cx="v.x" :cy="yl" r="30" fill="#24398A" />
        <text :x="v.x" :y="yl + 8" text-anchor="middle" style="fill: #fff; font-weight: 700; font-size: 22px">v{{ v.n }}</text>
        <text :x="v.x" :y="yl + 62" text-anchor="middle" style="fill: #1A2133; font-weight: 600; font-size: 20px">{{ v.fam }}</text>
        <text :x="v.x" :y="yl + 88" text-anchor="middle" style="fill: #5E6678; font-size: 18px">{{ v.cuando }} · AUC en prueba {{ v.auc.toFixed(3) }}</text>
      </g>
    </g>
    <g :style="{ opacity: p >= 2 ? 1 : 0.0, transition: 'opacity .5s' }">
      <circle :cx="xv3" :cy="yl" r="30" fill="#fff" stroke="#8DA3C9" stroke-width="2.5" stroke-dasharray="5 4" />
      <text :x="xv3" :y="yl + 8" text-anchor="middle" style="fill: #8DA3C9; font-weight: 700; font-size: 22px">v3</text>
      <text :x="xv3" :y="yl + 62" text-anchor="middle" style="fill: #5E6678; font-size: 18px">el próximo retador</text>
    </g>

    <!-- alias: banderas sobre un asta -->
    <g class="mov" :style="{ transform: `translateX(${xCampeon}px)` }">
      <line x1="0" x2="0" y1="104" :y2="yl - 30" stroke="#A87C12" stroke-width="3" />
      <rect x="0" y="98" width="164" height="42" rx="6" fill="#E1B73D" />
      <text x="82" y="126" text-anchor="middle" style="fill: #20305A; font-weight: 700; font-size: 20px">@champion</text>
    </g>
    <g :style="{ opacity: p === 1 ? 1 : 0, transition: 'opacity .5s' }">
      <line :x1="xv2" :x2="xv2" y1="164" :y2="yl - 30" stroke="#496EA5" stroke-width="3" />
      <rect :x="xv2" y="158" width="170" height="42" rx="6" fill="#fff" stroke="#496EA5" stroke-width="2.5" />
      <text :x="xv2 + 85" y="186" text-anchor="middle" style="fill: #24398A; font-weight: 700; font-size: 20px">@challenger</text>
    </g>
    <text v-if="p >= 2" :x="xv1" y="190" text-anchor="middle" style="fill: #5E6678; font-size: 18px">v1 se conserva:</text>
    <text v-if="p >= 2" :x="xv1" y="214" text-anchor="middle" style="fill: #5E6678; font-size: 18px">regresar = mover el alias</text>
  </svg>
</template>

<style scoped>
.mov { transition: transform .8s ease, x2 .8s ease; }
</style>
