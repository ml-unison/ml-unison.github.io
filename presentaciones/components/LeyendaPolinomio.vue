<script setup lang="ts">
withDefaults(defineProps<{ nuevos?: boolean, ajuste?: boolean, real?: boolean, residuos?: boolean }>(), {})
</script>

<template>
  <div class="leyenda">
    <span><i class="punto" /> datos que vio el modelo</span>
    <span v-if="nuevos"><i class="punto punto--nuevo" /> datos nuevos</span>
    <span v-if="ajuste"><i class="linea" /> curva aprendida</span>
    <span v-if="real"><i class="linea linea--real" /> la regla verdadera</span>
    <span v-if="residuos"><i class="linea linea--error" /> error en cada punto</span>
  </div>
</template>

<style scoped>
.leyenda { display: flex; gap: 34px; font-size: 19px; color: var(--tenue); }
.leyenda span { display: inline-flex; align-items: center; gap: 10px; }
.punto { width: 16px; height: 16px; border-radius: 50%; background: var(--dorado); display: inline-block; }
.punto--nuevo { background: #fff; border: 3px solid var(--marino); width: 14px; height: 14px; }
.linea { width: 30px; border-top: 4px solid var(--azul); display: inline-block; }
.linea--real { border-top: 3px dotted #8A91A0; }
.linea--error { width: 4px; height: 22px; border: 0; background: var(--rojo); }
</style>
