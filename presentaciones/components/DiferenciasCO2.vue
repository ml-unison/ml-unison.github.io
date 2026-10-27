<script setup lang="ts">
// El CO₂ de entrenamiento (1958–2014) original y diferenciado, con su ACF hasta 48 meses (rojo: múltiplos de 12).
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-02.json'
const D = (datos as any).diferencias
const props = withDefaults(defineProps<{ inicial?: string }>(), { inicial: 'y' })
const opciones = [
  { k: 'y', n: 'y', d: 'La ACF casi no baja: no estacionaria.' },
  { k: 'd1', n: '∇y', d: 'Sin tendencia, pero el ciclo anual domina.' },
  { k: 'd12', n: '∇₁₂y', d: 'Sin ciclo: el crecimiento anual, que todavía deambula.' },
  { k: 'd1d12', n: '∇∇₁₂y', d: 'Estacionaria: solo quedan las barras en 1 y en 12.' },
]
const sel = ref(props.inicial)
const S = computed(() => D[sel.value])
const op = computed(() => opciones.find(o => o.k === sel.value)!)
const W = 640, H = 330, m = { l: 62, r: 12, t: 16, b: 44 }
const x = scaleLinear().domain([1958, 2015]).range([m.l, W - m.r])
const y = computed(() => { const s = S.value.y as number[]; const a = Math.min(...s), b = Math.max(...s), h = (b - a) * 0.06; return scaleLinear().domain([a - h, b + h]).range([H - m.b, m.t]).nice() })
const camino = computed(() => line<number>().x((_, i) => x(S.value.t[i])).y(v => y.value(v))(S.value.y)!)
const f = (v: number) => String(v).replace('-', '−')
</script>

<template>
  <div class="dc">
    <div class="dc__arriba">
      <div class="dc__botones"><button v-for="o in opciones" :key="o.k" :class="{ activo: o.k === sel }" @click="sel = o.k">{{ o.n }}</button></div>
    </div>
    <div class="dc__graf">
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <g class="rejilla"><line v-for="t in y.ticks(4)" :key="t" :x1="m.l" :x2="W - m.r" :y1="y(t)" :y2="y(t)" /></g>
        <g class="eje">
          <text v-for="t in [1960, 1970, 1980, 1990, 2000, 2010]" :key="t" :x="x(t)" :y="H - m.b + 26" text-anchor="middle">{{ t }}</text>
          <text v-for="t in y.ticks(4)" :key="`y${t}`" :x="m.l - 10" :y="y(t) + 6" text-anchor="end">{{ f(t) }}</text>
        </g>
        <path :d="camino" fill="none" stroke="#24398A" :stroke-width="sel === 'y' ? 2 : 1.1" />
      </svg>
      <div>
        <CorrelogramaARIMA :valores="S.acf" :banda="S.banda" :periodo="12" titulo="ACF (rezagos en meses)" :ancho="470" :alto="290" />
        <div class="dc__desc">{{ op.d }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dc { display: flex; flex-direction: column; gap: 10px; }
.dc__botones { display: flex; gap: 8px; }
.dc__botones button { font: inherit; font-size: 18px; padding: 5px 14px; border: 1.5px solid var(--azul); border-radius: 6px; background: #fff; color: var(--azul); cursor: pointer; }
.dc__botones button.activo { background: var(--azul); color: #fff; }
.dc__graf { display: grid; grid-template-columns: 640px 470px; gap: 18px; align-items: start; }
.dc__desc { font-size: 20px; color: var(--tinta); margin-top: 6px; padding-left: 58px; }
</style>
