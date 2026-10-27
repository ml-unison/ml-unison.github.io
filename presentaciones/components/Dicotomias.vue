<script setup lang="ts">
// Las 8 dicotomías de 3 puntos, cada una con una recta que la produce; y 4 puntos que ninguna recta separa.
import datos from '../datos/u03.json'
const D = (datos as any).dicotomias
withDefaults(defineProps<{ xor?: boolean }>(), { xor: true })
const S = 130
const recta = (w: number[]) => {
  // w0 x + w1 y + b = 0, recortada al cuadro [0,1]²
  const [a, b, c] = w, pts: number[][] = []
  if (Math.abs(b) > 1e-9) for (const xx of [0, 1]) { const yy = -(a * xx + c) / b; if (yy >= 0 && yy <= 1) pts.push([xx, yy]) }
  if (Math.abs(a) > 1e-9) for (const yy of [0, 1]) { const xx = -(b * yy + c) / a; if (xx >= 0 && xx <= 1) pts.push([xx, yy]) }
  return pts.length >= 2 ? pts.slice(0, 2) : null
}
</script>

<template>
  <div class="di">
    <div class="di__ocho">
      <svg v-for="(r, k) in D.rectas" :key="k" :width="S" :height="S" viewBox="-0.05 -0.05 1.1 1.1" class="di__p">
        <line v-if="recta(r.w)" :x1="recta(r.w)![0][0]" :y1="1 - recta(r.w)![0][1]" :x2="recta(r.w)![1][0]" :y2="1 - recta(r.w)![1][1]" stroke="#C9971C" stroke-width="0.03" />
        <circle v-for="(p, i) in D.puntos" :key="i" :cx="p[0]" :cy="1 - p[1]" r="0.075" :fill="r.y[i] > 0 ? '#24398A' : '#fff'" :stroke="r.y[i] > 0 ? '#24398A' : '#D5433C'" stroke-width="0.035" />
      </svg>
    </div>
    <div v-if="xor" class="di__xor">
      <svg :width="190" :height="190" viewBox="-0.05 -0.05 1.1 1.1" class="di__p">
        <circle v-for="(p, i) in D.xor" :key="i" :cx="p[0]" :cy="1 - p[1]" r="0.07" :fill="p[2] > 0 ? '#24398A' : '#fff'" :stroke="p[2] > 0 ? '#24398A' : '#D5433C'" stroke-width="0.03" />
      </svg>
      <div class="di__t">Con 4 puntos, esta dicotomía no la produce ninguna recta.</div>
    </div>
  </div>
</template>

<style scoped>
.di { display: flex; gap: 48px; align-items: center; }
.di__ocho { display: grid; grid-template-columns: repeat(4, auto); gap: 14px; }
.di__p { background: #F6F7FA; border-radius: 8px; display: block; }
.di__xor { display: flex; flex-direction: column; gap: 10px; align-items: center; }
.di__t { font-size: 19px; color: var(--tenue); max-width: 220px; text-align: center; }
</style>
