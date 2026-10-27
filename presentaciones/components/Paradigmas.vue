<script setup lang="ts">
// Tres escenas pequeñas: supervisado, no supervisado y por refuerzo.
import { mulberry32, normal } from './azar'

const rnd = mulberry32(11)
const W = 320, H = 230
// Supervisado: dos clases separadas por una curva suave
const sup = Array.from({ length: 34 }, () => {
  const x = 20 + rnd() * 280, y = 20 + rnd() * 190
  const c = y < 115 + 45 * Math.sin(x / 60) ? 1 : 0
  return { x, y, c }
}).filter(p => Math.abs(p.y - (115 + 45 * Math.sin(p.x / 60))) > 12)
const frontera = Array.from({ length: 61 }, (_, i) => `${i ? 'L' : 'M'}${(i / 60) * W},${115 + 45 * Math.sin(((i / 60) * W) / 60)}`).join('')
// No supervisado: tres grupos sin etiqueta
const centros = [[80, 70], [235, 80], [160, 175]]
const nos = centros.flatMap(([cx, cy]) => Array.from({ length: 13 }, () => ({ x: cx + normal(rnd) * 22, y: cy + normal(rnd) * 17 })))
// Refuerzo: cuadrícula con un agente, una meta y un obstáculo
const celdas = Array.from({ length: 20 }, (_, i) => ({ x: 30 + (i % 5) * 52, y: 14 + Math.floor(i / 5) * 52 }))
const camino = 'M56,196 L56,144 L108,144 L160,144 L160,92 L212,92 L212,40 L264,40'
</script>

<template>
  <div class="paradigmas">
    <div class="paradigma">
      <svg :viewBox="`0 0 ${W} ${H}`">
        <path :d="frontera" fill="none" stroke="#C5CAD5" stroke-width="2" stroke-dasharray="6 6" />
        <circle v-for="(p, i) in sup" :key="i" :cx="p.x" :cy="p.y" r="7" :fill="p.c ? '#24398A' : '#E1B73D'" stroke="#fff" stroke-width="2" />
        <circle cx="168" cy="120" r="12" fill="#fff" stroke="#5E6678" stroke-width="2" stroke-dasharray="3 3" />
        <text x="168" y="127" text-anchor="middle" class="interroga">?</text>
      </svg>
      <h3>Supervisado</h3>
      <p>Ejemplos <b>con respuesta</b>. Aprender a predecir la respuesta de casos nuevos.</p>
      <p class="ej">Correo → ¿spam? · Lluvia → nivel de la presa</p>
    </div>
    <div class="paradigma">
      <svg :viewBox="`0 0 ${W} ${H}`">
        <ellipse v-for="(c, k) in centros" :key="`e${k}`" v-click="1" :cx="c[0]" :cy="c[1]" rx="62" ry="50" fill="#EFF2F9" stroke="#8DA3C9" stroke-width="1.5" />
        <circle v-for="(p, i) in nos" :key="i" :cx="p.x" :cy="p.y" r="6.5" fill="#6B7385" stroke="#fff" stroke-width="2" />
      </svg>
      <h3>No supervisado</h3>
      <p>Ejemplos <b>sin respuesta</b>. Descubrir estructura: grupos, direcciones, anomalías.</p>
      <p class="ej">Perfiles de clientes → segmentos</p>
    </div>
    <div class="paradigma">
      <svg :viewBox="`0 0 ${W} ${H}`">
        <rect v-for="(c, i) in celdas" :key="i" :x="c.x" :y="c.y" width="48" height="48" rx="6" fill="#F6F7FA" />
        <rect x="134" y="170" width="48" height="48" rx="6" fill="#FCEFEE" />
        <text x="158" y="202" text-anchor="middle" class="simbolo" fill="#D5433C">✕</text>
        <path :d="camino" fill="none" stroke="#24398A" stroke-width="3" stroke-dasharray="2 7" stroke-linecap="round" />
        <circle cx="56" cy="196" r="13" fill="#24398A" />
        <text x="264" y="49" text-anchor="middle" class="simbolo" fill="#C9971C">★</text>
      </svg>
      <h3>Por refuerzo</h3>
      <p>Un agente <b>actúa</b> y recibe recompensas. Aprender qué hacer en cada situación.</p>
      <p class="ej">Juegos, robots, control</p>
    </div>
  </div>
</template>

<style scoped>
.paradigmas { display: grid; grid-template-columns: repeat(3, 1fr); gap: 44px; margin-top: 6px; }
.paradigma svg { width: 100%; height: auto; display: block; margin-bottom: 18px; }
.paradigma h3 { font-family: var(--f-titulo); font-weight: 650; font-size: 26px; color: var(--marino); text-transform: none; letter-spacing: -0.01em; margin: 0 0 8px; }
.paradigma p { font-size: 21px; line-height: 1.4; color: var(--tinta); margin: 0 0 10px; }
.paradigma p.ej { font-size: 18px; color: var(--suave); }
.interroga { font: 700 16px var(--f-titulo); fill: #5E6678; }
.simbolo { font-size: 26px; }
</style>
