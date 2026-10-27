<script setup lang="ts">
// Ptolomeo (círculos sobre círculos) frente a Kepler (una elipse): la navaja de Occam en la historia.
const R = 128, r = 40, k = 7
const ptolomeo = Array.from({ length: 721 }, (_, i) => {
  const t = (i / 720) * 2 * Math.PI
  const x = R * Math.cos(t) + r * Math.cos(k * t), y = R * Math.sin(t) + r * Math.sin(k * t)
  return `${i ? 'L' : 'M'}${(x + 200).toFixed(1)},${(y + 190).toFixed(1)}`
}).join('')
const t0 = 0.55
const centroEpi = { x: 200 + R * Math.cos(t0), y: 190 + R * Math.sin(t0) }
const planeta = { x: centroEpi.x + r * Math.cos(k * t0), y: centroEpi.y + r * Math.sin(k * t0) }
// Elipse de Kepler con el Sol en un foco
const a = 160, b = 120, c = Math.sqrt(a * a - b * b)
</script>

<template>
  <div class="epi">
    <figure>
      <svg viewBox="0 0 400 380">
        <circle cx="200" cy="190" :r="R" fill="none" stroke="#C0CEE3" stroke-width="1.5" stroke-dasharray="5 6" />
        <path :d="ptolomeo" fill="none" stroke="#24398A" stroke-width="2.4" opacity="0.9" />
        <circle :cx="centroEpi.x" :cy="centroEpi.y" :r="r" fill="none" stroke="#8DA3C9" stroke-width="1.5" />
        <line x1="200" y1="190" :x2="centroEpi.x" :y2="centroEpi.y" stroke="#8DA3C9" stroke-width="1.5" />
        <line :x1="centroEpi.x" :y1="centroEpi.y" :x2="planeta.x" :y2="planeta.y" stroke="#8DA3C9" stroke-width="1.5" />
        <circle :cx="planeta.x" :cy="planeta.y" r="8" fill="#20305A" />
        <circle cx="200" cy="190" r="11" fill="#496EA5" />
        <text x="200" y="222" text-anchor="middle" class="et">Tierra</text>
      </svg>
      <figcaption><b>Ptolomeo</b> (siglo II): la Tierra al centro y círculos montados sobre círculos. Con suficientes epiciclos reproduce las observaciones.</figcaption>
    </figure>
    <figure v-click>
      <svg viewBox="0 0 400 380">
        <ellipse cx="200" cy="190" :rx="a" :ry="b" fill="none" stroke="#24398A" stroke-width="2.6" />
        <circle :cx="200 - c" cy="190" r="16" fill="#E1B73D" />
        <text :x="200 - c" y="228" text-anchor="middle" class="et">Sol</text>
        <circle :cx="200 + a * Math.cos(1.1)" :cy="190 - b * Math.sin(1.1)" r="8" fill="#20305A" />
      </svg>
      <figcaption><b>Kepler</b> (1609): el Sol en un foco y una sola elipse por planeta. Menos supuestos, y una ley que después explicó Newton.</figcaption>
    </figure>
  </div>
</template>

<style scoped>
.epi { display: grid; grid-template-columns: 1fr 1fr; gap: 56px; }
figure { margin: 0; }
svg { width: 100%; height: 330px; display: block; }
figcaption { font-size: 21px; line-height: 1.4; color: var(--tenue); margin-top: 6px; }
figcaption b { color: var(--marino); }
.et { font: 600 15px var(--f-texto); fill: #5E6678; }
</style>
