<script setup lang="ts">
// Fórmula KaTeX dentro de componentes (en Markdown basta con $…$).
import katex from 'katex'
import { computed } from 'vue'

const props = defineProps<{ t: string, display?: boolean }>()
const html = computed(() => katex.renderToString(props.t, { displayMode: !!props.display, throwOnError: false }))
</script>

<template><span v-html="html" /></template>
