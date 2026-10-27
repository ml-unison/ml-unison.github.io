<script setup lang="ts">
const props = defineProps<{ tipo?: string, nombre?: string, etiqueta?: string }>()
const nombres: Record<string, string> = {
  definicion: 'Definición', teorema: 'Teorema', proposicion: 'Proposición', algoritmo: 'Algoritmo',
  ejemplo: 'Ejemplo', idea: 'Idea clave', cuidado: 'Cuidado', pregunta: 'Para pensar', correcto: 'Correcto',
}
const tipo = props.tipo ?? 'definicion'
</script>

<template>
  <div class="caja" :class="`caja--${tipo}`">
    <div class="caja__etiqueta">{{ etiqueta ?? nombres[tipo] ?? tipo }}<span v-if="nombre"> · {{ nombre }}</span></div>
    <div class="caja__cuerpo"><slot /></div>
  </div>
</template>
