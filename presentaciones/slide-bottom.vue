<script setup lang="ts">
import { computed } from 'vue'
import { useNav, useSlideContext } from '@slidev/client'

const { $frontmatter, $page } = useSlideContext()
const { total } = useNav()
const { $slidev } = useSlideContext()
const visible = computed(() => !['portada', 'seccion'].includes($frontmatter.value?.layout ?? (($page.value === 1) ? 'portada' : 'default')))
const info = computed(() => $slidev.configs as Record<string, any>)
</script>

<template>
  <div v-if="visible" class="pie">
    <span><b>{{ info.unidad ?? 'Reconocimiento de Patrones' }}</b> · {{ info.title }}</span>
    <span>{{ $page }} / {{ total }}</span>
  </div>
</template>
