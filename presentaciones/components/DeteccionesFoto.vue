<script setup lang="ts">
// Una foto de Hermosillo con las detecciones de YOLO26n (confianza ≥ 0.25), calculadas en la libreta del
// ejemplo 5 (herramientas/laminas/_u06_04_deteccion.py). Cajas en pixeles de la foto reducida.
import datos from '../datos/u06-04.json'
const props = withDefaults(defineProps<{ foto?: number, etiquetas?: boolean }>(), { foto: 0, etiquetas: true })
const F = (datos as any).deteccion.fotos[props.foto]
const COL: Record<string, string> = { auto: '#24398A', persona: '#D5433C', autobús: '#C9971C' }
const col = (c: string) => COL[c] ?? '#20305A'
</script>

<template>
  <figure style="margin: 0">
    <svg class="grafica" :viewBox="`0 0 ${F.ancho} ${F.alto}`" style="border-radius: 8px">
      <image :href="F.imagen" x="0" y="0" :width="F.ancho" :height="F.alto" />
      <g v-for="(d, i) in F.detecciones" :key="i">
        <rect :x="d.caja[0]" :y="d.caja[1]" :width="d.caja[2] - d.caja[0]" :height="d.caja[3] - d.caja[1]"
              fill="none" :stroke="col(d.clase)" stroke-width="3" />
        <template v-if="props.etiquetas">
          <rect :x="d.caja[0]" :y="d.caja[1] - 20" :width="(d.clase.length + 5) * 8.4" height="20" :fill="col(d.clase)" />
          <text :x="d.caja[0] + 4" :y="d.caja[1] - 5" style="fill: #FFFFFF; font-size: 15px; font-weight: 600">{{ d.clase }} {{ d.conf.toFixed(2) }}</text>
        </template>
      </g>
    </svg>
    <figcaption style="font-size: 14px; color: var(--tenue); margin-top: 4px">
      {{ F.titulo }}. Foto: {{ F.autor }}, {{ F.licencia }}, Wikimedia Commons.
    </figcaption>
  </figure>
</template>
