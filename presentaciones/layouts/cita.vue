<script setup lang="ts">
defineProps<{ autor?: string, nota?: string }>()
</script>

<template>
  <div class="slidev-layout cita">
    <div class="cita__comillas">“</div>
    <slot />
    <div v-if="autor" class="cita__autor">{{ autor }}</div>
    <div v-if="nota" class="cita__nota">{{ nota }}</div>
  </div>
</template>
