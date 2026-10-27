<script setup lang="ts">
// Un botín de Fashion-MNIST y sus mapas con los filtros de Sobel vertical y horizontal (hechos a mano).
import datos from '../datos/u06-03.json'
const S = (datos as any).sobel
const lim = Math.max(...S.v.flat().map(Math.abs), ...S.h.flat().map(Math.abs))
const filtros = [
  { nombre: 'bordes verticales', K: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], mapa: S.v },
  { nombre: 'bordes horizontales', K: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], mapa: S.h },
]
const signo = (v: number) => (v < 0 ? `−${-v}` : String(v))
withDefaults(defineProps<{ tam?: number }>(), { tam: 250 })
</script>

<template>
  <div class="fs">
    <div class="fs__panel">
      <MapaPixeles :datos="S.prenda" :tam="tam" :vmax="255" />
      <div class="fs__t">botín (28 × 28)</div>
    </div>
    <div v-for="f in filtros" :key="f.nombre" class="fs__panel">
      <MapaPixeles :datos="f.mapa" :tam="tam * 26 / 28" modo="divergente" :vmax="lim" />
      <div class="fs__t">
        <span class="fs__k"><span v-for="(v, i) in f.K.flat()" :key="i" :class="{ pos: v > 0, neg: v < 0 }">{{ signo(v) }}</span></span>
        {{ f.nombre }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.fs { display: flex; gap: 34px; align-items: flex-start; }
.fs__panel { display: flex; flex-direction: column; align-items: center; gap: 10px; }
.fs__t { font-size: 19px; font-weight: 600; color: var(--marino); display: flex; align-items: center; gap: 12px; }
.fs__k { display: inline-grid; grid-template-columns: repeat(3, 26px); font-size: 14px; font-weight: 600; text-align: center; background: var(--fondo-alt); border-radius: 4px; padding: 2px; }
.fs__k .pos { color: #24398A; } .fs__k .neg { color: #D5433C; } .fs__k span { color: #8A91A0; }
.fs__k .pos, .fs__k .neg { font-weight: 700; }
</style>
