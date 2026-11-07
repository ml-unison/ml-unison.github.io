<script setup lang="ts">
// Tres formas de transferir una red preentrenada. Cada red se dibuja como capas (láminas) de izquierda a derecha:
// gris = congelada (pesos de ImageNet sin tocar), azul = se sigue entrenando, dorado = nueva.
const props = withDefaults(defineProps<{ hasta?: number }>(), { hasta: 3 })
const W = 1180, H = 380
const GRIS = '#8A91A0', AZUL = '#24398A', DORADO = '#C9971C'
const filas = [
  { t: 'atributos + regresión logística', capas: [GRIS, GRIS, GRIS, GRIS, GRIS], cab: 'logística (sklearn)', cabCol: DORADO },
  { t: 'sustituir solo la última capa', capas: [GRIS, GRIS, GRIS, GRIS, GRIS], cab: 'nn.Linear(512, 10)', cabCol: DORADO },
  { t: 'ajuste fino', capas: [AZUL, AZUL, AZUL, AZUL, AZUL], cab: 'nn.Linear(512, 10)', cabCol: DORADO },
]
const lado = [70, 62, 54, 46, 38]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <g v-for="(f, i) in filas" :key="i" v-show="i < props.hasta" :transform="`translate(0, ${30 + i * 120})`">
      <text x="0" y="45" style="fill: #1A2133; font-weight: 700; font-size: 26px">{{ f.t }}</text>
      <g v-for="(c, k) in f.capas" :key="k">
        <rect v-for="j in 3" :key="j" :x="400 + k * 95 + j * 5" :y="40 - lado[k] / 2 - j * 5 + 8" :width="lado[k]" :height="lado[k]"
              fill="#FFFFFF" :stroke="c" stroke-width="2" />
      </g>
      <text x="400" y="100" style="font-size: 19px">ResNet-18 de ImageNet (11 millones de parámetros)</text>
      <circle v-for="j in 4" :key="`o${j}`" cx="900" :cy="0 + j * 16" r="6" fill="#FFFFFF" :stroke="f.cabCol" stroke-width="2.2" />
      <text x="925" y="45" :style="`fill: #A87C12; font-weight: 700; font-size: 24px`">{{ f.cab }}</text>
      <line x1="860" y1="40" x2="888" y2="40" stroke="#8A91A0" stroke-width="1.5" />
    </g>
  </svg>
</template>
