<script setup lang="ts">
// Intersección sobre unión (IoU) de una caja verdadera (azul) y una predicha (dorada) que se desliza.
import { ref, computed } from 'vue'
const W = 560, H = 330
const real = { x1: 120, y1: 70, x2: 320, y2: 270 }
const d = ref(80)
const pred = computed(() => ({ x1: 120 + d.value, y1: 70 + d.value * 0.35, x2: 320 + d.value, y2: 270 + d.value * 0.35 }))
const inter = computed(() => {
  const p = pred.value
  const x1 = Math.max(real.x1, p.x1), y1 = Math.max(real.y1, p.y1)
  const x2 = Math.min(real.x2, p.x2), y2 = Math.min(real.y2, p.y2)
  return { x1, y1, x2, y2, a: Math.max(0, x2 - x1) * Math.max(0, y2 - y1) }
})
const area = (b: { x1: number, y1: number, x2: number, y2: number }) => (b.x2 - b.x1) * (b.y2 - b.y1)
const iou = computed(() => inter.value.a / (area(real) + area(pred.value) - inter.value.a))
</script>

<template>
  <div>
    <div class="deslizador">
      <span>desplazamiento</span>
      <input type="range" min="0" max="220" step="2" v-model.number="d" style="flex: 0 0 200px" />
      <span style="font-weight: 700; color: #24398A; white-space: nowrap">IoU = {{ iou.toFixed(2) }}</span>
    </div>
    <svg class="grafica" :viewBox="`0 0 ${W} ${H}`" style="background: #F6F7FA; border-radius: 8px">
      <rect v-if="inter.a > 0" :x="inter.x1" :y="inter.y1" :width="inter.x2 - inter.x1" :height="inter.y2 - inter.y1" fill="#C0CEE3" />
      <rect :x="real.x1" :y="real.y1" :width="real.x2 - real.x1" :height="real.y2 - real.y1" fill="none" stroke="#24398A" stroke-width="4" />
      <rect :x="pred.x1" :y="pred.y1" :width="pred.x2 - pred.x1" :height="pred.y2 - pred.y1" fill="none" stroke="#C9971C" stroke-width="4" stroke-dasharray="10 6" />
      <text :x="real.x1 + 8" :y="real.y1 + 26" style="fill: #24398A; font-weight: 700">verdadera</text>
      <text :x="Math.min(pred.x2, W - 6) - 8" :y="Math.min(pred.y2, H - 6) - 10" text-anchor="end" style="fill: #A87C12; font-weight: 700">predicha</text>
    </svg>
  </div>
</template>
