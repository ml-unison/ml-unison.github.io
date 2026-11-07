<script setup lang="ts">
// La idea de YOLO sobre una foto: rejilla de S × S; cada objeto lo predice la celda que contiene el centro
// de su caja. Se marcan esas celdas y los centros (detecciones de YOLO26n de la libreta del ejemplo 5).
import datos from '../datos/u06-04.json'
const props = withDefaults(defineProps<{ foto?: number }>(), { foto: 1 })
const D = (datos as any).deteccion
const F = D.fotos[props.foto]
const S = D.yolo.S as number
const cw = F.ancho / S, ch = F.alto / S
const objetos = F.detecciones.map((d: any) => {
  const cx = (d.caja[0] + d.caja[2]) / 2, cy = (d.caja[1] + d.caja[3]) / 2
  return { ...d, cx, cy, fila: Math.floor(cy / ch), col: Math.floor(cx / cw) }
})
const celdas = [...new Set(objetos.map((o: any) => `${o.fila},${o.col}`))].map((s: any) => s.split(',').map(Number))
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${F.ancho} ${F.alto}`" style="border-radius: 8px">
    <image :href="F.imagen" x="0" y="0" :width="F.ancho" :height="F.alto" opacity="0.75" />
    <rect v-for="(c, i) in celdas" :key="`c${i}`" :x="c[1] * cw" :y="c[0] * ch" :width="cw" :height="ch" fill="#E1B73D" fill-opacity="0.35" />
    <line v-for="k in S - 1" :key="`v${k}`" :x1="k * cw" :x2="k * cw" y1="0" :y2="F.alto" stroke="#FFFFFF" stroke-width="2" />
    <line v-for="k in S - 1" :key="`h${k}`" :y1="k * ch" :y2="k * ch" x1="0" :x2="F.ancho" stroke="#FFFFFF" stroke-width="2" />
    <g v-for="(o, i) in objetos" :key="i">
      <rect :x="o.caja[0]" :y="o.caja[1]" :width="o.caja[2] - o.caja[0]" :height="o.caja[3] - o.caja[1]" fill="none" stroke="#24398A" stroke-width="2.5" />
      <circle :cx="o.cx" :cy="o.cy" r="6" fill="#D5433C" stroke="#FFFFFF" stroke-width="2" />
    </g>
  </svg>
</template>
