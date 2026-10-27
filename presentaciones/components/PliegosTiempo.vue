<script setup lang="ts">
// Tres formas de partir 120 días: KFold aleatorio, origen móvil con ventana creciente y con ventana deslizante (y hueco).
import { scaleLinear } from 'd3-scale'
import datos from '../datos/u07-01.json'
const props = withDefaults(defineProps<{ mostrar?: string[], fila?: number }>(), { mostrar: () => ['kfold', 'creciente', 'deslizante'], fila: 28 })
const P = (datos as any).pliegos
const W = 1120, fila = props.fila, m = { l: 110, r: 10 }
const x = scaleLinear().domain([0, P.n]).range([m.l, W - m.r])
const dw = x(1) - x(0)
const titulos: Record<string, string> = {
  kfold: 'KFold aleatorio: el futuro se mezcla con el pasado',
  creciente: 'Origen móvil con ventana creciente',
  deslizante: 'Origen móvil con ventana deslizante (40 días) y hueco de 3 días',
}
const bloques = props.mostrar.map((k) => ({
  k, titulo: titulos[k],
  filas: Array.from({ length: 5 }, (_, f) => Array.from({ length: P.n }, (_, i) => {
    if (k === 'kfold') return P.kfold[i] === f ? 'p' : 'e'
    const [a, b, c, d] = P[k][f]
    return i >= c && i <= d ? 'p' : (i >= a && i <= b ? 'e' : '')
  })),
}))
const altoBloque = 5 * fila + 46
const H = bloques.length * altoBloque
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" role="img">
    <g v-for="(b, j) in bloques" :key="b.k" :transform="`translate(0, ${j * altoBloque})`">
      <text x="0" y="22" style="fill: #20305A; font-weight: 650; font-size: 20px">{{ b.titulo }}</text>
      <g v-for="(f, r) in b.filas" :key="r" :transform="`translate(0, ${34 + r * fila})`">
        <text x="0" :y="fila / 2 + 3" style="font-size: 17px">pliego {{ r + 1 }}</text>
        <template v-for="(c, i) in f" :key="i">
          <rect v-if="c" :x="x(i) + 0.5" y="0" :width="dw - 1.2" :height="fila - 6" :fill="c === 'p' ? '#D5433C' : '#24398A'" :opacity="c === 'p' ? 1 : 0.75" />
        </template>
      </g>
    </g>
  </svg>
</template>
