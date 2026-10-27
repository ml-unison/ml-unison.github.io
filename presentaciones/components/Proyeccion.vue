<script setup lang="ts">
// Mínimos cuadrados como proyección: y, su proyección ŷ sobre el espacio columna de X y el residuo perpendicular.
const O = [170, 300], A = [520, 300], B = [330, 190]          // plano en perspectiva: O, O+u, O+v
const P = [420, 248], Y = [420, 70]                          // ŷ en el plano; y arriba
const plano = `${O[0] - 90},${O[1] + 40} ${A[0] + 60},${A[1] + 40} ${B[0] + 260},${B[1] - 40} ${B[0] - 60},${B[1] - 40}`
</script>

<template>
  <svg class="grafica" viewBox="0 0 700 380">
    <polygon :points="plano" fill="#EFF2F9" stroke="#8DA3C9" stroke-width="1.5" />
    <text x="600" y="330" style="font-size: 18px; fill: #496EA5">espacio columna de X</text>
    <defs>
      <marker id="fl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="context-stroke" /></marker>
    </defs>
    <line :x1="O[0]" :y1="O[1]" :x2="A[0] - 140" :y2="A[1]" stroke="#6B7385" stroke-width="2.5" marker-end="url(#fl)" />
    <text :x="A[0] - 160" :y="A[1] + 30" style="font-size: 19px">columna 1</text>
    <line :x1="O[0]" :y1="O[1]" :x2="B[0]" :y2="B[1]" stroke="#6B7385" stroke-width="2.5" marker-end="url(#fl)" />
    <text :x="B[0] - 120" :y="B[1] - 6" style="font-size: 19px">columna 2</text>
    <line :x1="O[0]" :y1="O[1]" :x2="Y[0]" :y2="Y[1]" stroke="#24398A" stroke-width="3.5" marker-end="url(#fl)" />
    <text :x="Y[0] + 12" :y="Y[1] + 4" style="fill: #24398A; font-weight: 700; font-size: 22px">y</text>
    <line :x1="O[0]" :y1="O[1]" :x2="P[0]" :y2="P[1]" stroke="#C9971C" stroke-width="3.5" marker-end="url(#fl)" />
    <text :x="P[0] + 12" :y="P[1] + 24" style="fill: #1A2133; font-weight: 700; font-size: 22px">ŷ = Xw</text>
    <line :x1="Y[0]" :y1="Y[1] + 4" :x2="P[0]" :y2="P[1] - 4" stroke="#D5433C" stroke-width="3" stroke-dasharray="7 5" />
    <text :x="Y[0] + 14" :y="(Y[1] + P[1]) / 2" style="fill: #1A2133; font-weight: 600; font-size: 20px">y − ŷ</text>
    <path :d="`M${P[0]},${P[1] - 22} h16 v20`" fill="none" stroke="#1A2133" stroke-width="1.5" />
  </svg>
</template>
