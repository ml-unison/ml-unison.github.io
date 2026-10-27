<script setup lang="ts">
// Caso real: una cifra grande a la izquierda, la historia a la derecha.
defineProps<{ cifra?: string, cifraNota?: string, fuente?: string, caso?: string }>()
</script>

<template>
  <div class="slidev-layout caso">
    <div class="caso__izq">
      <div class="caso__etiqueta">{{ caso ?? 'Caso real' }}</div>
      <div class="caso__cifra" v-html="cifra" />
      <div class="caso__cifranota">{{ cifraNota }}</div>
    </div>
    <div class="caso__der">
      <slot />
      <div v-if="fuente" class="caso__fuente">Fuente: {{ fuente }}</div>
    </div>
  </div>
</template>
