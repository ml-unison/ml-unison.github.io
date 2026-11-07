<script setup lang="ts">
// Exactitud top-1 en ImageNet contra millones de parámetros. Parámetros contados con torchvision (sin pesos);
// exactitud de los pesos IMAGENET1K_V1 de torchvision, salvo Xception (Chollet, 2017). Datos: u06_03.py.
import { scaleLog, scaleLinear } from 'd3-scale'
import datos from '../datos/u06-03.json'
const A = (datos as any).arquitecturas as { nombre: string, anio: number, params_m: number, top1: number, fuente: string }[]
const W = 620, H = 400, m = { l: 64, r: 20, t: 16, b: 66 }
const x = scaleLog().domain([4, 200]).range([m.l, W - m.r])
const y = scaleLinear().domain([54, 81]).range([H - m.b, m.t])
// posición de cada etiqueta (para que no se encimen)
const pos: Record<string, [number, number, string]> = {
  'AlexNet': [-12, 6, 'end'], 'VGG-16': [-12, 6, 'end'], 'GoogLeNet': [12, 20, 'start'],
  'ResNet-18': [12, 6, 'start'], 'ResNet-50': [12, 10, 'start'], 'Inception V3': [-12, 22, 'end'],
  'Xception': [-12, -8, 'end'], 'EfficientNet-B0': [12, -10, 'start'],
}
const color = (n: string) => n === 'EfficientNet-B0' || n === 'Xception' ? '#C9971C' : n === 'AlexNet' || n === 'VGG-16' ? '#8A91A0' : '#24398A'
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="[5, 10, 20, 50, 100, 200]" :y-ticks="[55, 60, 65, 70, 75, 80]"
          x-titulo="millones de parámetros (escala logarítmica)" y-titulo="exactitud top-1 (%)" />
    <g v-for="a in A" :key="a.nombre">
      <circle :cx="x(a.params_m)" :cy="y(a.top1)" r="8" :fill="color(a.nombre)" />
      <text :x="x(a.params_m) + (pos[a.nombre]?.[0] ?? 12)" :y="y(a.top1) + (pos[a.nombre]?.[1] ?? 6)"
            :text-anchor="pos[a.nombre]?.[2] ?? 'start'" style="fill: #1A2133; font-weight: 600">{{ a.nombre }}</text>
    </g>
  </svg>
</template>
