<script setup lang="ts">
// merge por la clave de la presa; con `duplicado`, el catálogo trae una clave repetida.
import { computed } from 'vue'
const props = withDefaults(defineProps<{ duplicado?: boolean, tam?: number }>(), { tam: 19 })
const registros = [['PECSO', '2022-09-15', 2805.8], ['LCDSO', '2022-09-15', 627.1], ['PECSO', '2022-09-16', 2807.8]]
const catalogo = computed(() => props.duplicado
  ? [['PECSO', 'El Novillo', 2833], ['LCDSO', 'La Angostura', 703], ['PECSO', 'Plutarco Elías Calles', 2833]]
  : [['PECSO', 'El Novillo', 2833], ['LCDSO', 'La Angostura', 703]])
const resultado = computed(() => registros.flatMap(r => catalogo.value.filter(c => c[0] === r[0]).map(c => [r[0], r[1], r[2], c[1], c[2]])))
const malas = computed(() => props.duplicado ? resultado.value.map((f, i) => (f[0] === 'PECSO' ? i : -1)).filter(i => i >= 0) : [])
</script>

<template>
  <div class="union">
    <div class="union__fila">
      <div>
        <div class="union__titulo">registros ({{ registros.length }} filas)</div>
        <TablaMini :columnas="['clave', 'fecha', 'almac.']" :filas="registros" :clave="0" :tam="tam" />
      </div>
      <div class="union__op">+</div>
      <div>
        <div class="union__titulo">catálogo</div>
        <TablaMini :columnas="['clave', 'presa', 'cap_namo']" :filas="catalogo" :clave="0" :tam="tam" :malas="duplicado ? [0, 2] : []" />
      </div>
    </div>
    <div class="union__flecha"><code>registros.merge(catalogo, on="clave")</code></div>
    <div>
      <div class="union__titulo" :class="{ mal: duplicado }">resultado ({{ resultado.length }} filas)</div>
      <TablaMini :columnas="['clave', 'fecha', 'almac.', 'presa', 'cap_namo']" :filas="resultado" :clave="0" :tam="tam" :malas="malas" />
    </div>
  </div>
</template>

<style scoped>
.union { display: flex; flex-direction: column; align-items: center; gap: 14px; }
.union__fila { display: flex; align-items: center; gap: 30px; }
.union__op { font: 400 40px var(--f-titulo); color: var(--azul-claro); }
.union__titulo { font: 650 14px var(--f-titulo); letter-spacing: .1em; text-transform: uppercase; color: var(--suave); margin-bottom: 8px; }
.union__titulo.mal { color: #B23028; }
.union__flecha { font-size: 17px; color: var(--tenue); }
.union__flecha::before { content: '↓'; display: block; text-align: center; font-size: 26px; color: var(--azul-claro); }
</style>
