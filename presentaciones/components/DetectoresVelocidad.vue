<script setup lang="ts">
// Velocidad contra exactitud en PASCAL VOC 2007 (mAP con IoU ≥ 0.5), según los artículos originales:
// R-CNN con VGG-16: 66.0 % y 47 s por imagen (Girshick, «Fast R-CNN», 2015, tabla 4);
// Fast R-CNN 70.0 % a 0.5 imágenes/s, Faster R-CNN (VGG-16) 73.2 % a 7/s, YOLO 63.4 % a 45/s y
// Fast YOLO 52.7 % a 155/s (Redmon et al., 2016, tabla 1). Fast R-CNN incluye el tiempo de las propuestas.
import { scaleLog, scaleLinear } from 'd3-scale'
const puntos = [
  { n: 'R-CNN', a: 2014, fps: 1 / 47, map: 66.0, dx: 12, dy: 6 },
  { n: 'Fast R-CNN', a: 2015, fps: 0.5, map: 70.0, dx: 12, dy: -10 },
  { n: 'Faster R-CNN', a: 2015, fps: 7, map: 73.2, dx: -12, dy: -12, fin: true },
  { n: 'YOLO', a: 2016, fps: 45, map: 63.4, dx: -12, dy: 24, fin: true },
  { n: 'Fast YOLO', a: 2016, fps: 155, map: 52.7, dx: -12, dy: 6, fin: true },
]
const W = 640, H = 380, m = { l: 70, r: 20, t: 16, b: 66 }
const x = scaleLog().domain([0.01, 300]).range([m.l, W - m.r])
const y = scaleLinear().domain([50, 76]).range([H - m.b, m.t])
const xt = [0.01, 0.1, 1, 10, 100]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <Ejes :x="x" :y="y" :x-ticks="xt" :y-ticks="[50, 55, 60, 65, 70, 75]" :x-formato="(v: number) => String(v)"
          x-titulo="imágenes por segundo (escala logarítmica)" y-titulo="mAP en VOC 2007 (%)" />
    <rect :x="x(30)" :y="m.t" :width="W - m.r - x(30)" :height="H - m.b - m.t" fill="#FBF6E6" />
    <text :x="x(30) + 8" :y="m.t + 22" style="fill: #A87C12; font-weight: 600; font-size: 16px">tiempo real</text>
    <g v-for="p in puntos" :key="p.n">
      <circle :cx="x(p.fps)" :cy="y(p.map)" r="8" :fill="p.n.includes('YOLO') ? '#C9971C' : '#24398A'" />
      <text :x="x(p.fps) + p.dx" :y="y(p.map) + p.dy" :text-anchor="p.fin ? 'end' : 'start'" style="fill: #1A2133; font-weight: 600">{{ p.n }} ({{ p.a }})</text>
    </g>
  </svg>
</template>
