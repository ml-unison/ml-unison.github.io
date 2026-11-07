<script setup lang="ts">
// La red convolucional de Fashion-MNIST partida en dos: la parte convolucional (preprocesamiento aprendido
// que convierte 784 pixeles en 1,568 atributos locales) y el clasificador de encima.
// Con `paso` se muestra solo la primera parte (0), las dos (1). Datos de la prenda: u06-03.json (sobel.prenda).
import datos from '../datos/u06-03.json'
const props = withDefaults(defineProps<{ paso?: number }>(), { paso: 1 })
const prenda = (datos as any).sobel.prenda as number[][]
const W = 1180, H = 380, cy = 200
const P = 4.2                                    // pixeles de dibujo por pixel de la prenda
const px0 = 10
// etapas: canales, lado; se dibujan como láminas apiladas
const etapas = [
  { c: 16, d: 28, x: 175, et: 'conv 3×3 + ReLU' },
  { c: 16, d: 14, x: 330, et: 'max-pool' },
  { c: 32, d: 14, x: 450, et: 'conv 3×3 + ReLU' },
  { c: 32, d: 7, x: 600, et: 'max-pool' },
]
const lamina = (e: { c: number, d: number, x: number }) => {
  const n = e.c === 16 ? 6 : 10, lado = e.d * 3.6, off = 6
  return Array.from({ length: n }, (_, k) => ({
    x: e.x + k * off, y: cy - lado / 2 - k * off + (n * off) / 2, lado,
  }))
}
const vx = 745, vh = 200
const celdas = Array.from({ length: 46 }, (_, k) => k)
const ocultas = Array.from({ length: 6 }, (_, k) => cy - 100 + k * 40)
const salidas = Array.from({ length: 10 }, (_, k) => cy - 108 + k * 24)
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <!-- fondos de las dos partes -->
    <rect x="160" y="8" :width="vx + 30 - 160" :height="H - 16" rx="14" fill="#EFF2F9" />
    <text :x="(160 + vx + 30) / 2" y="40" text-anchor="middle" style="fill: #24398A; font-weight: 700; font-size: 22px">
      preprocesamiento aprendido: atributos locales
    </text>
    <g v-if="props.paso >= 1">
      <rect :x="vx + 50" y="8" :width="W - vx - 58" :height="H - 16" rx="14" fill="#FBF6E6" />
      <text :x="(vx + 50 + W) / 2" y="40" text-anchor="middle" style="fill: #A87C12; font-weight: 700; font-size: 22px">
        clasificador
      </text>
    </g>

    <!-- la prenda -->
    <g :transform="`translate(${px0}, ${cy - 14 * P})`">
      <rect x="-2" y="-2" :width="28 * P + 4" :height="28 * P + 4" fill="#F6F7FA" rx="4" />
      <template v-for="(fila, r) in prenda" :key="r">
        <rect v-for="(v, c) in fila" :key="c" v-show="v > 8" :x="c * P" :y="r * P" :width="P + 0.3" :height="P + 0.3"
              :fill="`rgba(32,48,90,${v / 255})`" />
      </template>
    </g>
    <text :x="px0 + 14 * P" :y="cy + 14 * P + 30" text-anchor="middle">784 pixeles</text>

    <!-- etapas convolucionales -->
    <g v-for="(e, j) in etapas" :key="j">
      <rect v-for="(l, k) in lamina(e)" :key="k" :x="l.x" :y="l.y" :width="l.lado" :height="l.lado"
            fill="#FFFFFF" stroke="#24398A" stroke-width="1.4" />
      <text :x="e.x + (e.c === 16 ? 15 : 27) + e.d * 1.8" :y="cy + 85" text-anchor="middle" style="fill: #1A2133; font-weight: 600">
        {{ e.c }}×{{ e.d }}×{{ e.d }}
      </text>
      <text :x="e.x + (e.c === 16 ? 15 : 27) + e.d * 1.8" :y="cy + 110" text-anchor="middle" style="font-size: 16px">{{ e.et }}</text>
    </g>

    <!-- vector de atributos -->
    <rect v-for="k in celdas" :key="`v${k}`" :x="vx" :y="cy - vh / 2 + k * (vh / celdas.length)" width="18"
          :height="vh / celdas.length - 1" :fill="k % 3 === 0 ? '#496EA5' : k % 3 === 1 ? '#8DA3C9' : '#C0CEE3'" />
    <text :x="vx + 9" :y="cy + vh / 2 + 28" text-anchor="middle" style="fill: #1A2133; font-weight: 600">1,568</text>
    <text :x="vx + 9" :y="cy + vh / 2 + 50" text-anchor="middle" style="font-size: 16px">atributos</text>

    <!-- clasificador -->
    <g v-if="props.paso >= 1">
      <line v-for="(yo, i) in ocultas" :key="`a${i}`" :x1="vx + 18" :y1="cy - vh / 2 + i * 45 + 10" :x2="900" :y2="yo"
            stroke="#C9971C" stroke-opacity="0.35" />
      <template v-for="(yo, i) in ocultas" :key="`b${i}`">
        <line v-for="(ys, j) in salidas" :key="j" x1="900" :y1="yo" x2="1030" :y2="ys" stroke="#C9971C" stroke-opacity="0.25" />
      </template>
      <circle v-for="(yo, i) in ocultas" :key="`o${i}`" cx="900" :cy="yo" r="12" fill="#FFFFFF" stroke="#C9971C" stroke-width="2" />
      <circle v-for="(ys, j) in salidas" :key="`s${j}`" cx="1030" :cy="ys" r="9" :fill="j === 9 ? '#24398A' : '#FFFFFF'"
              stroke="#24398A" stroke-width="1.6" />
      <text x="1048" :y="salidas[9] + 6" style="fill: #24398A; font-weight: 700">botín</text>
      <text x="965" :y="cy + vh / 2 + 28" text-anchor="middle" style="fill: #1A2133; font-weight: 600">red multicapa o GLM</text>
      <text x="965" :y="cy + vh / 2 + 50" text-anchor="middle" style="font-size: 16px">softmax con 10 clases</text>
    </g>
  </svg>
</template>
