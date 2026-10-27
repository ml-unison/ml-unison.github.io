<script setup lang="ts">
defineProps<{ si?: boolean }>()
</script>

<template><span class="veredicto" :class="si ? 'veredicto--si' : 'veredicto--no'"><slot>{{ si ? 'aprender' : 'programar' }}</slot></span></template>
