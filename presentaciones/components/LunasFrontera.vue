<script setup lang="ts">
// Red 2-16-2 (tanh) en las lunas: la frontera de decisión en varias épocas y la curva de entrenamiento.
// Con `comparar`, dos paneles: la misma red sin activación (lineal) y con tanh. Datos: herramientas/laminas/u06_01.py.
import { ref, computed } from 'vue'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-01.json'
withDefaults(defineProps<{ comparar?: boolean }>(), { comparar: false })
const L = (datos as any).lunas
const W = 520, H = 385
const x = scaleLinear().domain([-2, 3]).range([0, W]), y = scaleLinear().domain([-1.6, 2.1]).range([H, 0])
const sel = ref(L.epocas.length - 1)
// curva de pérdida
const CW = 520, CH = 250, cm = { l: 58, r: 14, t: 18, b: 56 }
const cx = scaleLog().domain([1, 300]).range([cm.l, CW - cm.r]), cy = scaleLinear().domain([0.1, 0.4]).range([CH - cm.b, cm.t])
const camino = line<number>().x((_, i) => cx(i + 1)).y(v => cy(Math.min(v, 0.4)))(L.perdida)!
const epocaSel = computed(() => L.epocas[sel.value])
</script>

<template>
  <div v-if="comparar" class="lf2">
    <div v-for="(reg, k) in [L.lineal, L.regiones[L.regiones.length - 1]]" :key="k">
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" style="border-radius: 8px">
        <Regiones :datos="reg" :x="x" :y="y" :colores="['#C0CEE3', '#F3C7C4']" :opacidad="0.8" />
        <circle v-for="(v, i) in L.x" :key="i" :cx="x(v)" :cy="y(L.y[i])" r="3.6" :fill="L.g[i] ? '#D5433C' : '#24398A'" stroke="#fff" stroke-width="0.6" />
      </svg>
      <div class="lf__t">{{ k === 0 ? `σ(z) = z · exactitud ${L.exac_lineal.toFixed(3)}` : `tanh · exactitud ${L.exac_red.toFixed(3)}` }}</div>
    </div>
  </div>
  <div v-else class="lf">
    <div>
      <div class="lf__botones"><button v-for="(e, k) in L.epocas" :key="e" :class="{ activo: k === sel }" @click="sel = k">época {{ e }}</button></div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" style="border-radius: 8px">
        <Regiones :datos="L.regiones[sel]" :x="x" :y="y" :colores="['#C0CEE3', '#F3C7C4']" :opacidad="0.8" />
        <circle v-for="(v, i) in L.x" :key="i" :cx="x(v)" :cy="y(L.y[i])" r="3.6" :fill="L.g[i] ? '#D5433C' : '#24398A'" stroke="#fff" stroke-width="0.6" />
      </svg>
    </div>
    <div>
      <svg class="grafica" :viewBox="`0 0 ${CW} ${CH}`">
        <Ejes :x="cx" :y="cy" :x-ticks="[1, 3, 10, 30, 100, 300]" :y-ticks="[0.1, 0.2, 0.3, 0.4]" x-titulo="época (escala logarítmica)" y-titulo="entropía cruzada" />
        <line :x1="cm.l" :x2="CW - cm.r" :y1="cy(L.perdida_log)" :y2="cy(L.perdida_log)" stroke="#6B7385" stroke-width="1.5" stroke-dasharray="6 5" />
        <text :x="CW - cm.r" :y="cy(L.perdida_log) - 8" text-anchor="end" style="font-size: 16px">regresión logística</text>
        <path :d="camino" fill="none" stroke="#24398A" stroke-width="2.5" />
        <line v-if="epocaSel > 0" :x1="cx(epocaSel)" :x2="cx(epocaSel)" :y1="cm.t" :y2="CH - cm.b" stroke="#C9971C" stroke-width="2.5" />
      </svg>
      <slot />
    </div>
  </div>
</template>

<style scoped>
.lf { display: grid; grid-template-columns: 520px 1fr; gap: 34px; align-items: start; }
.lf2 { display: grid; grid-template-columns: 1fr 1fr; gap: 34px; }
.lf__t { text-align: center; font-weight: 600; font-size: 19px; color: var(--marino); margin-top: 6px; }
.lf__botones { display: flex; gap: 8px; margin-bottom: 10px; }
.lf__botones button { font: inherit; font-size: 17px; padding: 4px 12px; border: 1.5px solid var(--azul); border-radius: 6px; background: #fff; color: var(--azul); cursor: pointer; }
.lf__botones button.activo { background: var(--azul); color: #fff; }
</style>
