<script setup lang="ts">
defineProps<{ curso?: string, unidad?: string, subtitulo?: string, autor?: string, afiliacion?: string, fecha?: string }>()
</script>

<template>
  <div class="slidev-layout portada">
    <div class="portada__texto">
      <div class="portada__curso">{{ curso ?? 'Reconocimiento de Patrones' }} <b v-if="unidad">· {{ unidad }}</b></div>
      <slot />
      <div v-if="subtitulo" class="portada__subtitulo">{{ subtitulo }}</div>
      <div class="portada__regla" />
      <div class="portada__autor">
        <b>{{ autor ?? 'Julio Waissman' }}</b><br>
        <span>{{ afiliacion ?? 'Departamento de Matemáticas' }}<template v-if="fecha"> · {{ fecha }}</template></span>
      </div>
      <div class="portada__marca">
        <img src="/marca/letragrama-unison.png" alt="Universidad de Sonora">
        <div>Licenciatura en<br>Ciencias de la Computación</div>
      </div>
    </div>
    <div class="portada__figura"><MotivoFrontera /></div>
  </div>
</template>
