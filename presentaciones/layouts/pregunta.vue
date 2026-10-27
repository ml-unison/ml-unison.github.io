<script setup lang="ts">
defineProps<{ etiqueta?: string }>()
</script>

<template>
  <div class="slidev-layout pregunta">
    <div class="pregunta__etiqueta">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 9a4 4 0 1 1 5.5 3.7c-.9.4-1.5 1.2-1.5 2.2V16M12 20h.01" /></svg>
      {{ etiqueta ?? 'Pregunta para el grupo' }}
    </div>
    <slot />
  </div>
</template>
