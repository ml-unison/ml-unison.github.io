<script setup lang="ts">
// Pingüinos (largo × profundidad del pico): la partición en rectángulos de un árbol de decisión
// según su profundidad máxima. Los rectángulos son las hojas exactas del árbol de scikit-learn.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { SERIE } from './colores'
import datos from '../datos/u05-01.json'
const props = withDefaults(defineProps<{ prof?: string, botones?: boolean, ancho?: number, alto?: number }>(),
  { prof: '1', botones: true, ancho: 640, alto: 460 })
const P = (datos as any).pinguinos
const niveles = P.niveles as any[]
const sel = ref(props.prof)
const N = computed(() => niveles.find(n => n.prof === sel.value))
const W = props.ancho, H = props.alto, m = { l: 64, r: 12, t: 30, b: 64 }
const x = scaleLinear().domain([31, 61]).range([m.l, W - m.r]), y = scaleLinear().domain([12.5, 22]).range([H - m.b, m.t])
const TINTAS = ['#C0CEE3', '#F3C7C4', '#EFDDAA']
const nombres = [{ t: 'Adelie', x: 33, y: 21.2 }, { t: 'Chinstrap', x: 50.5, y: 21.2 }, { t: 'Gentoo', x: 52, y: 13.1 }]
</script>

<template>
  <div class="ap" :class="{ 'ap--solo': !botones }">
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <rect v-for="(q, k) in N.rect" :key="`${sel}-${k}`" :x="x(q[0])" :y="y(q[3])" :width="x(q[1]) - x(q[0])" :height="y(q[2]) - y(q[3])"
            :fill="TINTAS[q[4]]" stroke="#fff" stroke-width="1.5" />
      <Ejes :x="x" :y="y" :x-ticks="[35, 40, 45, 50, 55, 60]" :y-ticks="[14, 16, 18, 20, 22]" x-titulo="largo del pico (mm)" y-titulo="profundidad del pico (mm)" />
      <circle v-for="(v, i) in P.x" :key="i" :cx="x(v)" :cy="y(P.y[i])" r="3.6" :fill="SERIE[P.g[i]]" stroke="#fff" stroke-width="0.7" />
      <text v-for="n in nombres" :key="n.t" :x="x(n.x)" :y="y(n.y)" style="fill: #1A2133; font-weight: 700">{{ n.t }}</text>
    </svg>
    <div v-if="botones" class="ap__lado">
      <div class="ap__etiqueta">profundidad máxima</div>
      <div class="ap__botones"><button v-for="n in niveles" :key="n.prof" :class="{ activo: n.prof === sel }" @click="sel = n.prof">{{ n.prof }}</button></div>
      <table>
        <tr><td>hojas (rectángulos)</td><td><b>{{ N.hojas }}</b></td></tr>
        <tr><td>exactitud de entrenamiento</td><td><b>{{ N.exact.toFixed(3) }}</b></td></tr>
        <tr><td>exactitud de validación cruzada</td><td><b>{{ N.cv.toFixed(3) }}</b></td></tr>
      </table>
      <slot />
    </div>
  </div>
</template>

<style scoped>
.ap { display: grid; grid-template-columns: 640px 1fr; gap: 36px; align-items: start; }
.ap--solo { display: block; }
.ap__lado { display: flex; flex-direction: column; gap: 14px; font-size: 20px; }
.ap__etiqueta { font: 600 15px var(--f-titulo); letter-spacing: .08em; text-transform: uppercase; color: var(--tenue); }
.ap__botones { display: flex; gap: 8px; flex-wrap: wrap; }
.ap__botones button { font: inherit; font-size: 18px; padding: 6px 14px; border: 1.5px solid var(--azul); border-radius: 6px; background: #fff; color: var(--azul); cursor: pointer; }
.ap__botones button.activo { background: var(--azul); color: #fff; }
.ap table { font-size: 20px; }
.ap table td { padding: 8px 10px 8px 0; }
.ap table b { color: var(--azul); font-variant-numeric: tabular-nums; }
</style>
