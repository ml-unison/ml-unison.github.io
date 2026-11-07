<script setup lang="ts">
// Dos convoluciones de 3×3 apiladas ven lo mismo que una de 5×5 (vista de lado, en una dimensión).
// A la izquierda, la pila de VGG; a la derecha, una sola capa de 5×5. Se resalta lo que ve la neurona de arriba.
const C = 46, G = 6
const fila = (n: number, y: number, x0: number, activos: number[]) =>
  Array.from({ length: n }, (_, i) => ({ x: x0 + i * (C + G), y, on: activos.includes(i) }))
const W = 870, H = 340
const izq = { x0: 20 }
const der = { x0: 470 }
// pila: entrada de 7, capa 1 de 5, capa 2 de 3; la neurona central de arriba ve 5 de la entrada
const e1 = fila(7, 250, izq.x0, [1, 2, 3, 4, 5])
const c1 = fila(5, 150, izq.x0 + (C + G), [1, 2, 3])
const c2 = fila(3, 50, izq.x0 + 2 * (C + G), [1])
const e2 = fila(7, 250, der.x0, [1, 2, 3, 4, 5])
const d1 = fila(3, 50, der.x0 + 2 * (C + G), [1])
const centro = (c: { x: number, y: number }) => ({ x: c.x + C / 2, y: c.y + C / 2 })
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <!-- conexiones -->
    <g stroke="#24398A" stroke-opacity="0.35" stroke-width="1.6">
      <template v-for="(a, i) in c1" :key="`a${i}`">
        <line v-if="a.on" v-for="k in [0, 1, 2]" :key="k" :x1="centro(a).x" :y1="a.y + C" :x2="centro(e1[i + k]).x" :y2="e1[i + k].y" />
      </template>
      <line v-for="k in [0, 1, 2]" :key="`b${k}`" :x1="centro(c2[1]).x" :y1="c2[1].y + C" :x2="centro(c1[1 + k]).x" :y2="c1[1 + k].y" />
      <line v-for="k in [0, 1, 2, 3, 4]" :key="`c${k}`" :x1="centro(d1[1]).x" :y1="d1[1].y + C" :x2="centro(e2[1 + k]).x" :y2="e2[1 + k].y" stroke="#C9971C" stroke-opacity="0.6" />
    </g>
    <g v-for="(f, j) in [e1, c1, c2]" :key="`i${j}`">
      <rect v-for="(c, i) in f" :key="i" :x="c.x" :y="c.y" :width="C" :height="C" rx="5"
            :fill="c.on ? '#C0CEE3' : '#F6F7FA'" :stroke="c.on ? '#24398A' : '#E2E5EC'" stroke-width="1.6" />
    </g>
    <g v-for="(f, j) in [e2, d1]" :key="`d${j}`">
      <rect v-for="(c, i) in f" :key="i" :x="c.x" :y="c.y" :width="C" :height="C" rx="5"
            :fill="c.on ? '#FBF6E6' : '#F6F7FA'" :stroke="c.on ? '#C9971C' : '#E2E5EC'" stroke-width="1.6" />
    </g>
    <text :x="izq.x0 + 3.5 * (C + G)" :y="H - 6" text-anchor="middle" style="fill: #24398A; font-weight: 700; font-size: 26px">dos capas de 3×3</text>
    <text :x="der.x0 + 3.5 * (C + G)" :y="H - 6" text-anchor="middle" style="fill: #A87C12; font-weight: 700; font-size: 26px">una capa de 5×5</text>
    <text x="455" y="282" text-anchor="middle" style="font-size: 20px">entrada</text>
  </svg>
</template>
