<script setup lang="ts">
// Ejemplo numérico de la correlación cruzada: imagen 5×5, filtro 3×3, salida 3×3.
// Los botones (o un clic en la salida) mueven la ventana; abajo se ve la suma de productos.
import { ref, computed } from 'vue'
import datos from '../datos/u06-03.json'
const C = (datos as any).conv
const X: number[][] = C.X, K: number[][] = C.K, Y: number[][] = C.Y
const p = ref(0)
const i = computed(() => Math.floor(p.value / 3)), j = computed(() => p.value % 3)
const t = 52, sep = 64
const ox = 0, kx = 5 * t + sep, yx = kx + 3 * t + sep
const W = yx + 3 * t + 4, H = 5 * t + 4
const signo = (v: number) => (v < 0 ? `−${-v}` : String(v))
const terminos = computed(() => {
  const s: string[] = []
  for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) if (K[u][v] !== 0 && X[i.value + u][j.value + v] !== 0) s.push(`(${signo(K[u][v])})·${X[i.value + u][j.value + v]}`)
  return s
})
</script>

<template>
  <div class="cv">
    <svg :viewBox="`-2 -34 ${W} ${H + 36}`" class="grafica">
      <text :x="ox" y="-12" class="cv__t">imagen X</text>
      <text :x="kx" y="-12" class="cv__t">filtro K</text>
      <text :x="yx" y="-12" class="cv__t">salida Y</text>
      <g v-for="(fila, a) in X" :key="`x${a}`">
        <g v-for="(v, b) in fila" :key="b">
          <rect :x="ox + b * t" :y="a * t" :width="t" :height="t" :fill="v ? '#C0CEE3' : '#F6F7FA'" stroke="#fff" stroke-width="2" />
          <text :x="ox + b * t + t / 2" :y="a * t + t / 2 + 8" text-anchor="middle" class="cv__n">{{ v }}</text>
        </g>
      </g>
      <rect :x="ox + j * t" :y="i * t" :width="3 * t" :height="3 * t" fill="none" stroke="#C9971C" stroke-width="4" rx="3" />
      <g v-for="(fila, a) in K" :key="`k${a}`">
        <g v-for="(v, b) in fila" :key="b">
          <rect :x="kx + b * t" :y="a * t" :width="t" :height="t" :fill="v > 0 ? '#24398A' : v < 0 ? '#D5433C' : '#F6F7FA'" :opacity="v ? 0.85 : 1" stroke="#fff" stroke-width="2" />
          <text :x="kx + b * t + t / 2" :y="a * t + t / 2 + 8" text-anchor="middle" class="cv__n" :style="{ fill: v ? '#fff' : '#1A2133' }">{{ signo(v) }}</text>
        </g>
      </g>
      <text :x="kx - sep / 2" :y="1.5 * t + 10" text-anchor="middle" class="cv__op">⋆</text>
      <text :x="yx - sep / 2" :y="1.5 * t + 10" text-anchor="middle" class="cv__op">=</text>
      <g v-for="(fila, a) in Y" :key="`y${a}`">
        <g v-for="(v, b) in fila" :key="b" style="cursor: pointer" @click="p = 3 * a + b">
          <rect :x="yx + b * t" :y="a * t" :width="t" :height="t" :fill="3 * a + b <= p ? '#EFDDAA' : '#F6F7FA'" stroke="#fff" stroke-width="2" />
          <text v-if="3 * a + b <= p" :x="yx + b * t + t / 2" :y="a * t + t / 2 + 8" text-anchor="middle" class="cv__n">{{ v }}</text>
        </g>
      </g>
      <rect :x="yx + j * t" :y="i * t" :width="t" :height="t" fill="none" stroke="#C9971C" stroke-width="4" rx="3" />
    </svg>
    <div class="cv__abajo">
      <button @click="p = (p + 8) % 9">◀</button>
      <button @click="p = (p + 1) % 9">▶</button>
      <span>Y[{{ i }}, {{ j }}] = <template v-if="terminos.length">{{ terminos.join(' + ') }}</template><template v-else>0</template> = <b>{{ Y[i][j] }}</b></span>
    </div>
  </div>
</template>

<style scoped>
.cv { display: flex; flex-direction: column; gap: 14px; }
.cv .grafica .cv__n { font-size: 24px; fill: #1A2133; font-weight: 600; font-variant-numeric: tabular-nums; }
.cv .grafica .cv__t { font-size: 19px; fill: #5E6678; font-weight: 600; }
.cv .grafica .cv__op { font-size: 40px; fill: #20305A; }
.cv__abajo { display: flex; align-items: center; gap: 10px; font-size: 21px; }
.cv__abajo b { color: var(--azul); }
.cv__abajo span { margin-left: 10px; }
.cv__abajo button { font: inherit; font-size: 16px; padding: 3px 12px; border: 1.5px solid var(--azul); border-radius: 6px; background: #fff; color: var(--azul); cursor: pointer; }
</style>
