<script setup lang="ts">
const pasos = [
  ['Definir el', 'problema y', 'la métrica'],
  ['Obtener y', 'entender', 'los datos'],
  ['Separar', 'entrenamiento', 'y prueba'],
  ['Preprocesar', 'y crear', 'atributos'],
  ['Entrenar y', 'seleccionar', 'modelos'],
  ['Evaluar en', 'prueba', '(una vez)'],
  ['Desplegar', 'y', 'monitorear'],
]
const tipo = ['', '', 'oro', '', 'clave', 'oro', '']
const W = 1128, ancho = 140, sep = (W - 7 * ancho) / 6, yc = 76, alto = 116
const cx = (i: number) => i * (ancho + sep) + ancho / 2
const arco = (a: number, b: number, prof: number) => {
  const y0 = yc + alto / 2
  return `M${cx(a)},${y0 + 4} C${cx(a)},${y0 + prof} ${cx(b)},${y0 + prof} ${cx(b)},${y0 + 10}`
}
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} 330`" role="img" aria-label="Ciclo de vida de un modelo de aprendizaje">
    <defs>
      <marker id="punta" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#20305A" />
      </marker>
      <marker id="punta-tenue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#496EA5" />
      </marker>
    </defs>
    <g v-for="(p, i) in pasos" :key="i">
      <rect :x="cx(i) - ancho / 2" :y="yc - alto / 2" :width="ancho" :height="alto" rx="10"
            :fill="tipo[i] === 'clave' ? '#24398A' : tipo[i] === 'oro' ? '#FBF6E6' : '#fff'"
            :stroke="tipo[i] === 'oro' ? '#E1B73D' : '#24398A'" stroke-width="2" />
      <text :x="cx(i)" :y="yc - alto / 2 - 14" text-anchor="middle" style="font-family: var(--f-titulo); font-weight: 650; font-size: 14px; fill: #8A91A0">{{ i + 1 }}</text>
      <text :x="cx(i)" :y="yc - 16" text-anchor="middle" :style="{ fill: tipo[i] === 'clave' ? '#fff' : '#20305A', fontWeight: 600, fontSize: '17px' }">
        <tspan v-for="(t, k) in p" :key="k" :x="cx(i)" :dy="k ? 23 : 0">{{ t }}</tspan>
      </text>
      <line v-if="i < 6" :x1="cx(i) + ancho / 2 + 4" :x2="cx(i + 1) - ancho / 2 - 6" :y1="yc" :y2="yc" stroke="#20305A" stroke-width="2.2" marker-end="url(#punta)" />
    </g>
    <path :d="arco(4, 3, 70)" fill="none" stroke="#496EA5" stroke-width="2.2" stroke-dasharray="6 6" marker-end="url(#punta-tenue)" />
    <text :x="(cx(3) + cx(4)) / 2" :y="yc + alto / 2 + 74" text-anchor="middle" style="font-size:17px; fill:#496EA5">validación cruzada</text>
    <path :d="arco(6, 1, 170)" fill="none" stroke="#496EA5" stroke-width="2.2" stroke-dasharray="6 6" marker-end="url(#punta-tenue)" />
    <text :x="(cx(1) + cx(6)) / 2" :y="yc + alto / 2 + 156" text-anchor="middle" style="font-size:17px; fill:#496EA5">datos nuevos, <tspan font-style="italic">drift</tspan>: reentrenar</text>
  </svg>
</template>
