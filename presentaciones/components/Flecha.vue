<script setup lang="ts">
withDefaults(defineProps<{ largo?: number, punteada?: boolean }>(), { largo: 46, punteada: false })
</script>

<template>
  <svg class="flecha" :style="{ width: `${largo}px` }" :viewBox="`0 0 ${largo} 16`" aria-hidden="true">
    <line x1="4" y1="8" :x2="largo - 10" y2="8" stroke="currentColor" stroke-width="2.4" :stroke-dasharray="punteada ? '5 5' : undefined" />
    <path :d="`M${largo - 12},2 L${largo - 2},8 L${largo - 12},14`" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round" />
  </svg>
</template>
