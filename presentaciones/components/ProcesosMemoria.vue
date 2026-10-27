<script setup lang="ts">
// Cinco procesos simulados (500 observaciones; se dibujan 200): la serie, su ACF y su PACF. Botones para cambiar de proceso.
import { ref, computed } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u07-02.json'
const D = (datos as any)
const opciones = [
  { k: 'ruido', n: 'ruido blanco', d: 'Sin memoria: nada sale de la banda.' },
  { k: 'ar09', n: 'AR(1), φ = 0.9', d: 'Rachas largas; la ACF decae como 0.9ᵏ; la PACF se corta en 1.' },
  { k: 'ar07', n: 'AR(1), φ = −0.7', d: 'Zigzag; la ACF alterna de signo; la PACF se corta en 1.' },
  { k: 'ma08', n: 'MA(1), θ = 0.8', d: 'Un solo eco: la ACF se corta en 1 (ρ₁ = 0.49); la PACF decae.' },
  { k: 'caminata', n: 'caminata aleatoria', d: 'Deambula; la ACF casi no baja: no es estacionaria.' },
]
const sel = ref('ar09')
const P = computed(() => D.procesos[sel.value])
const op = computed(() => opciones.find(o => o.k === sel.value)!)
const W = 1128, H = 190, m = { l: 58, r: 14, t: 12, b: 14 }
const x = scaleLinear().domain([0, 199]).range([m.l, W - m.r])
const y = computed(() => { const s = P.value.serie as number[]; const a = Math.min(...s), b = Math.max(...s), h = (b - a) * 0.06; return scaleLinear().domain([a - h, b + h]).range([H - m.b, m.t]) })
const camino = computed(() => line<number>().x((_, i) => x(i)).y(v => y.value(v))(P.value.serie)!)
</script>

<template>
  <div class="pm">
    <div class="pm__arriba">
      <div class="pm__botones"><button v-for="o in opciones" :key="o.k" :class="{ activo: o.k === sel }" @click="sel = o.k">{{ o.n }}</button></div>
      <div class="pm__desc">{{ op.d }}</div>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
      <line :x1="m.l" :x2="W - m.r" :y1="y(0)" :y2="y(0)" stroke="#E2E5EC" stroke-width="1.5" />
      <path :d="camino" fill="none" stroke="#24398A" stroke-width="2" stroke-linejoin="round" />
    </svg>
    <div class="pm__abajo">
      <CorrelogramaARIMA :valores="P.acf" :banda="D.banda" :teorica="P.teorica" titulo="ACF (puntos: teórica)" :ancho="550" :alto="230" />
      <CorrelogramaARIMA :valores="P.pacf" :banda="D.banda" titulo="PACF" :ancho="550" :alto="230" />
    </div>
  </div>
</template>

<style scoped>
.pm { display: flex; flex-direction: column; gap: 6px; }
.pm__arriba { display: flex; align-items: center; gap: 24px; }
.pm__botones { display: flex; gap: 8px; flex-wrap: nowrap; }
.pm__botones button { font: inherit; font-size: 17px; padding: 5px 12px; border: 1.5px solid var(--azul); border-radius: 6px; background: #fff; color: var(--azul); cursor: pointer; white-space: nowrap; }
.pm__botones button.activo { background: var(--azul); color: #fff; }
.pm__desc { font-size: 19px; color: var(--tinta); line-height: 1.3; }
.pm__abajo { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
</style>
