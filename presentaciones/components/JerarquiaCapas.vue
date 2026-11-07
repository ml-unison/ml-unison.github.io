<script setup lang="ts">
// Qué activa a cada capa de una VGG-16 de ImageNet: para cuatro canales de cinco capas, los 9 recortes de fotos
// de Imagenette que más activan al canal (tipo = 'parches', a la manera de Zeiler y Fergus, 2014) o la imagen
// que lo maximiza, obtenida por ascenso por gradiente (tipo = 'maxima'). Datos: herramientas/laminas/u06_04.py.
import datos from '../datos/u06-04.json'
const props = withDefaults(defineProps<{ tipo?: 'parches' | 'maxima' }>(), { tipo: 'parches' })
const capas = (datos as any).capas as { nombre: string, campo: number, lado: number, canales: { canal: number, parches: string, maxima: string }[] }[]
const desc = ['bordes y colores', 'curvas y esquinas', 'texturas y patrones', 'partes de objetos', 'objetos']
</script>

<template>
  <div class="jc">
    <div v-for="(c, i) in capas" :key="c.nombre" class="jc__capa">
      <div class="jc__rejilla">
        <img v-for="k in c.canales" :key="k.canal" :src="props.tipo === 'parches' ? k.parches : k.maxima" :alt="`${c.nombre}, canal ${k.canal}`" />
      </div>
      <div class="jc__nombre">{{ c.nombre }}</div>
      <div class="jc__desc">{{ desc[i] }}<span v-if="props.tipo === 'parches'"> · ve {{ c.campo }} px</span></div>
    </div>
  </div>
</template>

<style scoped>
.jc { display: grid; grid-template-columns: repeat(5, 1fr); gap: 18px; }
.jc__rejilla { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
.jc__rejilla img { width: 100%; display: block; border-radius: 3px; image-rendering: auto; }
.jc__nombre { font-family: var(--f-codigo); font-size: 18px; color: var(--marino); font-weight: 600; margin-top: 8px; text-align: center; }
.jc__desc { font-size: 16px; color: var(--tenue); text-align: center; }
</style>
