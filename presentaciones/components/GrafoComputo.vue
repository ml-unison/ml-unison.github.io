<script setup lang="ts">
// Grafo de cómputo de una neurona logística, ℓ = −ln σ(w x + b), con x = 1.5, y = 1.
// paso 0: solo el grafo; 1: valores hacia adelante (azul); 2: adjuntos ∂ℓ/∂v hacia atrás (rojo).
// Datos: herramientas/laminas/u06_01.py (mismos números que la libreta).
import datos from '../datos/u06-01.json'
withDefaults(defineProps<{ paso?: number }>(), { paso: 2 })
const G = (datos as any).grafo
const f = (v: number, d = 3) => String(+v.toFixed(d)).replace('-', '−')
const W = 1180, H = 340, R = 36
const nodos = [
  { id: 'mul', x: 270, y: 150, op: '×' },
  { id: 'sum', x: 510, y: 150, op: '+' },
  { id: 'sig', x: 750, y: 150, op: 'σ' },
  { id: 'ln', x: 990, y: 150, op: '−ln' },
]
const hojas = [
  { id: 'w', x: 70, y: 70, n: 'w', v: G.w, g: G.barw, param: true },
  { id: 'x', x: 70, y: 240, n: 'x', v: G.x, g: null, param: false },
  { id: 'b', x: 380, y: 285, n: 'b', v: G.b, g: G.barb, param: true },
]
// aristas: de (x1, y1) a (x2, y2), con el valor que llevan y su adjunto
const P = Object.fromEntries([...nodos, ...hojas].map(n => [n.id, n]))
const aristas = [
  { de: 'mul', a: 'sum', n: 'u', v: G.u, g: G.baru },
  { de: 'sum', a: 'sig', n: 'z', v: G.z, g: G.barz },
  { de: 'sig', a: 'ln', n: 'ŷ', v: G.yg, g: G.baryg },
]
const recorta = (a: any, b: any, ra: number, rb: number) => {
  const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy)
  return { x1: a.x + dx / d * ra, y1: a.y + dy / d * ra, x2: b.x - dx / d * rb, y2: b.y - dy / d * rb }
}
const hojaArista = (h: string, n: string) => recorta(P[h], P[n], 30, R + 6)
const salida = { x1: P.ln.x + R + 4, y1: P.ln.y, x2: W - 40, y2: P.ln.y }
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <defs>
      <marker id="gc-gris" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8A91A0" /></marker>
      <marker id="gc-rojo" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#D5433C" /></marker>
    </defs>

    <!-- aristas de las hojas -->
    <line v-bind="hojaArista('w', 'mul')" stroke="#8A91A0" stroke-width="2.5" marker-end="url(#gc-gris)" />
    <line v-bind="hojaArista('x', 'mul')" stroke="#8A91A0" stroke-width="2.5" marker-end="url(#gc-gris)" />
    <line v-bind="hojaArista('b', 'sum')" stroke="#8A91A0" stroke-width="2.5" marker-end="url(#gc-gris)" />
    <!-- aristas entre operaciones -->
    <g v-for="a in aristas" :key="a.n">
      <line v-bind="recorta(P[a.de], P[a.a], R + 4, R + 6)" stroke="#8A91A0" stroke-width="2.5" marker-end="url(#gc-gris)" />
      <text :x="(P[a.de].x + P[a.a].x) / 2" :y="P[a.de].y - 18" text-anchor="middle" class="gc-n">{{ a.n }}</text>
      <text v-if="paso >= 1" :x="(P[a.de].x + P[a.a].x) / 2" :y="P[a.de].y - 46" text-anchor="middle" class="gc-v">{{ f(a.v) }}</text>
      <g v-if="paso >= 2">
        <line :x1="P[a.a].x - R - 10" :x2="P[a.de].x + R + 12" :y1="P[a.de].y + 22" :y2="P[a.de].y + 22" stroke="#D5433C" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#gc-rojo)" />
        <text :x="(P[a.de].x + P[a.a].x) / 2" :y="P[a.de].y + 52" text-anchor="middle" class="gc-g">∂ℓ/∂{{ a.n }} = {{ f(a.g) }}</text>
      </g>
    </g>
    <!-- salida ℓ -->
    <line v-bind="salida" stroke="#8A91A0" stroke-width="2.5" marker-end="url(#gc-gris)" />
    <text :x="W - 34" :y="P.ln.y + 8" class="gc-n" style="font-size: 26px">ℓ</text>
    <text v-if="paso >= 1" :x="(salida.x1 + salida.x2) / 2" :y="P.ln.y - 18" text-anchor="middle" class="gc-v">{{ f(G.l) }}</text>
    <text v-if="paso >= 2" :x="(salida.x1 + salida.x2) / 2" :y="P.ln.y + 52" text-anchor="middle" class="gc-g">∂ℓ/∂ℓ = 1</text>

    <!-- operaciones -->
    <g v-for="n in nodos" :key="n.id">
      <circle :cx="n.x" :cy="n.y" :r="R" fill="#fff" stroke="#20305A" stroke-width="2.5" />
      <text :x="n.x" :y="n.y + 10" text-anchor="middle" class="gc-op">{{ n.op }}</text>
    </g>
    <!-- hojas -->
    <g v-for="h in hojas" :key="h.id">
      <circle :cx="h.x" :cy="h.y" r="26" :fill="h.param ? '#E8ECF5' : '#F1F1F1'" :stroke="h.param ? '#24398A' : '#8A91A0'" stroke-width="2" />
      <text :x="h.x" :y="h.y + 9" text-anchor="middle" class="gc-n" style="font-size: 26px">{{ h.n }}</text>
      <text v-if="paso >= 1 || !h.param" :x="h.x + (h.id === 'b' ? 38 : 0)" :y="h.id === 'w' ? h.y - 38 : h.id === 'x' ? h.y + 52 : h.y + 9"
            :text-anchor="h.id === 'b' ? 'start' : 'middle'" class="gc-v">{{ f(h.v) }}</text>
      <text v-if="paso >= 2 && h.g !== null" :x="h.id === 'w' ? h.x + 40 : h.x + 38" :y="h.id === 'w' ? h.y - 34 : h.y + 40"
            text-anchor="start" class="gc-g">∂ℓ/∂{{ h.n }} = {{ f(h.g) }}</text>
    </g>
  </svg>
</template>

<style scoped>
.gc-op { font: 700 26px var(--f-titulo); fill: #20305A; }
.grafica .gc-n { font-style: italic; font-size: 22px; fill: #1A2133; font-weight: 600; }
.grafica .gc-v { fill: #24398A; font-weight: 700; font-size: 21px; }
.grafica .gc-g { fill: #D5433C; font-weight: 700; font-size: 21px; }
</style>
