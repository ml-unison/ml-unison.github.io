<script setup lang="ts">
// Ciclo de vida de un modelo, dibujado como un anillo de pasos.
const pasos = [
  ['Definir el problema', 'y la métrica de éxito'],
  ['Obtener y explorar', 'los datos'],
  ['Separar', 'entrenamiento y prueba'],
  ['Preparar', 'los atributos'],
  ['Entrenar', 'y validar modelos'],
  ['Evaluar en prueba', 'una sola vez'],
  ['Desplegar', 'y monitorear'],
]
const clave = [2, 5]
const cx = 560, cy = 228, R = 172
const ang = (i: number) => -Math.PI / 2 + (i / pasos.length) * 2 * Math.PI
const pos = (i: number, rr = R) => ({ x: cx + rr * Math.cos(ang(i)), y: cy + rr * Math.sin(ang(i)) })
const arco = (i: number) => {
  const a0 = ang(i) + 0.2, a1 = ang(i + 1) - 0.2
  return `M${cx + R * Math.cos(a0)},${cy + R * Math.sin(a0)} A${R},${R} 0 0 1 ${cx + R * Math.cos(a1)},${cy + R * Math.sin(a1)}`
}
const ancla = (i: number) => {
  const c = Math.cos(ang(i))
  return Math.abs(c) < 0.2 ? 'middle' : c > 0 ? 'start' : 'end'
}
</script>

<template>
  <svg class="grafica" viewBox="0 0 1120 456" role="img" aria-label="Ciclo de vida de un modelo de aprendizaje">
    <defs>
      <marker id="punta-anillo" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L10,5 L0,10 z" fill="#8DA3C9" />
      </marker>
    </defs>
    <path v-for="(_, i) in pasos" :key="`a${i}`" :d="arco(i)" fill="none" stroke="#8DA3C9" stroke-width="2.5" marker-end="url(#punta-anillo)" />
    <g v-for="(p, i) in pasos" :key="i">
      <circle :cx="pos(i).x" :cy="pos(i).y" r="24" :fill="clave.includes(i) ? '#E1B73D' : '#24398A'" />
      <text :x="pos(i).x" :y="pos(i).y + 7" text-anchor="middle" class="num" :style="{ fill: clave.includes(i) ? `#20305A` : `#fff` }">{{ i + 1 }}</text>
      <text :x="pos(i, R + 44).x" :y="pos(i, R + 44).y - 2" :text-anchor="ancla(i)" class="paso">
        <tspan>{{ p[0] }}</tspan>
        <tspan :x="pos(i, R + 44).x" dy="24" class="paso2">{{ p[1] }}</tspan>
      </text>
    </g>
    <text :x="cx" :y="cy - 6" text-anchor="middle" class="centro">se itera</text>
    <text :x="cx" :y="cy + 22" text-anchor="middle" class="centro2">muchas veces</text>
  </svg>
</template>

<style scoped>
.num { font: 700 20px var(--f-titulo); }
.paso { font: 650 20px var(--f-titulo); fill: #20305A; }
.paso2 { font: 400 18px var(--f-texto); fill: #5E6678; }
.centro { font: 600 22px var(--f-titulo); fill: #8DA3C9; }
.centro2 { font: 400 19px var(--f-texto); fill: #8DA3C9; }
</style>
