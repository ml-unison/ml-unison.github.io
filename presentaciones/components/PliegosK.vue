<script setup lang="ts">
// Validación cruzada con k particiones: en cada ronda, una parte valida y las demás entrenan.
withDefaults(defineProps<{ k?: number, ancho?: number }>(), { k: 5, ancho: 620 })
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${ancho} ${k * 40 + 50}`" role="img">
    <g v-for="i in k" :key="i" :transform="`translate(0, ${(i - 1) * 40})`">
      <text x="0" y="22" style="font-size: 17px">ronda {{ i }}</text>
      <rect v-for="j in k" :key="j" :x="90 + (j - 1) * ((ancho - 220) / k)" y="4" :width="(ancho - 220) / k - 5" height="24" rx="4"
            :fill="i === j ? '#C9971C' : '#C0CEE3'" />
      <text :x="ancho - 120" y="22" style="font-size: 17px; fill: #1A2133">e<tspan dy="5" style="font-size: 13px">{{ i }}</tspan></text>
    </g>
    <rect x="90" :y="k * 40 + 14" width="18" height="14" rx="3" fill="#C0CEE3" /><text x="114" :y="k * 40 + 27" style="font-size: 16px">entrena</text>
    <rect x="200" :y="k * 40 + 14" width="18" height="14" rx="3" fill="#C9971C" /><text x="224" :y="k * 40 + 27" style="font-size: 16px">valida</text>
  </svg>
</template>
