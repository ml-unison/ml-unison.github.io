<script setup lang="ts">
// Un dígito escrito a mano (8×8, intensidades 0–16) dibujado como cuadrícula de tinta.
const fid = `tinta-${Math.random().toString(36).slice(2)}`
withDefaults(defineProps<{ px: number[], tam?: number, color?: string, marco?: boolean }>(), { tam: 56, color: '#20305A', marco: false })
</script>

<template>
  <svg :width="tam" :height="tam" viewBox="0 0 8 8" class="digito" :class="{ 'digito--marco': marco }" aria-hidden="true">
    <defs><filter :id="fid" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="0.16" /></filter></defs>
    <g :filter="`url(#${fid})`"><rect v-for="(v, i) in px" :key="i" v-show="v > 0" :x="i % 8" :y="Math.floor(i / 8)" width="1" height="1"
          :fill="color" :opacity="Math.min(1, v / 16 * 1.08)" /></g>
  </svg>
</template>

<style scoped>
.digito { display: block; }
.digito--marco { background: #F6F7FA; border-radius: 8px; padding: 6px; box-sizing: content-box; }
</style>
