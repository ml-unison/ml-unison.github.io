<script setup lang="ts">
// Una urna con proporción μ de canicas rojas y una muestra de N canicas con fracción ν.
import { mulberry32 } from './azar'
const r = mulberry32(11)
const canicas = Array.from({ length: 140 }, (_, i) => ({ x: 40 + r() * 220, y: 90 + r() * 200, roja: r() < 0.4 }))
  .filter(c => Math.hypot((c.x - 150) / 120, (c.y - 190) / 110) < 1)
const muestra = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
</script>

<template>
  <svg class="grafica" viewBox="0 0 640 340" role="img">
    <path d="M60,70 Q40,190 70,300 Q150,330 230,300 Q260,190 240,70 Z" fill="#F6F7FA" stroke="#8A91A0" stroke-width="2.5" />
    <rect x="80" y="55" width="140" height="18" rx="6" fill="#E2E5EC" />
    <circle v-for="(c, i) in canicas" :key="i" :cx="c.x" :cy="c.y" r="7.5" :fill="c.roja ? '#D5433C' : '#24398A'" opacity="0.9" />
    <text x="150" y="30" text-anchor="middle" class="etiqueta" style="fill: #1A2133">urna: μ = fracción roja</text>
    <path d="M270,190 C320,170 340,170 380,175" fill="none" stroke="#8A91A0" stroke-width="2" marker-end="url(#fl)" />
    <defs><marker id="fl" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0L10,5L0,10z" fill="#8A91A0" /></marker></defs>
    <circle v-for="(m, i) in muestra" :key="`m${i}`" :cx="410 + (i % 5) * 44" :cy="155 + Math.floor(i / 5) * 44" r="16" :fill="m ? '#D5433C' : '#24398A'" />
    <text x="498" y="270" text-anchor="middle" class="etiqueta" style="fill: #1A2133">muestra: ν = 3/10</text>
  </svg>
</template>
