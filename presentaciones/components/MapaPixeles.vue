<script setup lang="ts">
// Una matriz (imagen en grises, filtro o mapa de activación) dibujada pixel por pixel en un <canvas>.
// modo «gris»: 0 = blanco, vmax = tinta marina. modo «divergente»: rojo negativo, azul positivo, centrado en 0.
// El contenido del slot se dibuja encima, en un SVG con las coordenadas de los pixeles (viewBox = columnas × filas).
import { ref, onMounted, watch, computed } from 'vue'

const props = withDefaults(defineProps<{
  datos: number[][], tam?: number, modo?: 'gris' | 'divergente', vmax?: number, marco?: boolean, rejilla?: boolean,
}>(), { tam: 200, modo: 'gris', marco: true, rejilla: false })

const lienzo = ref<HTMLCanvasElement | null>(null)
const filas = computed(() => props.datos.length)
const cols = computed(() => props.datos[0].length)
const alto = computed(() => props.tam * filas.value / cols.value)

const mezcla = (a: number[], b: number[], t: number) => a.map((v, i) => Math.round(v + (b[i] - v) * t))
const BLANCO = [255, 255, 255], MARINO = [32, 48, 90], AZUL = [36, 57, 138], ROJO = [213, 67, 60], CENTRO = [247, 247, 247]

function dibujar() {
  const c = lienzo.value
  if (!c) return
  const ctx = c.getContext('2d')!
  const img = ctx.createImageData(cols.value, filas.value)
  const todos = props.datos.flat()
  const vmax = props.vmax ?? (props.modo === 'gris' ? Math.max(...todos) : Math.max(...todos.map(Math.abs))) ?? 1
  todos.forEach((v, k) => {
    let rgb: number[]
    if (props.modo === 'gris') rgb = mezcla(BLANCO, MARINO, Math.max(0, Math.min(1, v / (vmax || 1))))
    else {
      const t = Math.max(-1, Math.min(1, v / (vmax || 1)))
      rgb = t >= 0 ? mezcla(CENTRO, AZUL, t) : mezcla(CENTRO, ROJO, -t)
    }
    img.data.set([...rgb, 255], 4 * k)
  })
  ctx.putImageData(img, 0, 0)
}
onMounted(dibujar)
watch(() => props.datos, dibujar)
</script>

<template>
  <div class="mp" :class="{ 'mp--marco': marco }" :style="{ width: `${tam}px`, height: `${alto}px` }">
    <canvas ref="lienzo" :width="cols" :height="filas" :style="{ width: `${tam}px`, height: `${alto}px` }" />
    <svg class="mp__capa" :viewBox="`0 0 ${cols} ${filas}`" :width="tam" :height="alto" preserveAspectRatio="none">
      <g v-if="rejilla" stroke="#fff" stroke-width="0.04">
        <line v-for="i in cols - 1" :key="`c${i}`" :x1="i" :x2="i" y1="0" :y2="filas" />
        <line v-for="i in filas - 1" :key="`f${i}`" :y1="i" :y2="i" x1="0" :x2="cols" />
      </g>
      <slot />
    </svg>
  </div>
</template>

<style scoped>
.mp { position: relative; flex: none; }
.mp canvas { display: block; image-rendering: pixelated; }
.mp--marco { outline: 1px solid var(--linea); }
.mp__capa { position: absolute; left: 0; top: 0; overflow: visible; }
</style>
