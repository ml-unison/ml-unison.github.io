<script setup lang="ts">
// Una "tabla de memoria": ejemplos vistos con su respuesta, y un caso nuevo sin respuesta.
import datos from '../datos/u01.json'
const D = datos.digitos as any
const vistos = [[D.sietes[2], '7'], [D.unos[2], '1'], [D.mosaico[1][3], '3'], [D.sietes[3], '7']]
const nuevo = D.sietes[11]
</script>

<template>
  <div class="memo">
    <div class="memo__titulo">Tabla en memoria</div>
    <div v-for="(v, i) in vistos" :key="i" class="memo__fila">
      <Digito :px="v[0]" :tam="58" />
      <span class="memo__flecha">→</span>
      <span class="memo__resp">{{ v[1] }}</span>
    </div>
    <div class="memo__fila memo__fila--nuevo">
      <Digito :px="nuevo" :tam="58" color="#D5433C" />
      <span class="memo__flecha">→</span>
      <span class="memo__resp memo__resp--no">?</span>
      <span class="memo__nota">imagen nueva: no está en la tabla</span>
    </div>
  </div>
</template>

<style scoped>
.memo { background: var(--fondo-alt); border-radius: 14px; padding: 22px 30px; }
.memo__titulo { font: 650 14px var(--f-titulo); letter-spacing: .12em; text-transform: uppercase; color: var(--suave); margin-bottom: 10px; }
.memo__fila { display: flex; align-items: center; gap: 18px; padding: 6px 0; border-bottom: 1px solid var(--linea); }
.memo__fila--nuevo { border-bottom: 0; margin-top: 10px; }
.memo__flecha { color: var(--azul-claro); font-size: 26px; }
.memo__resp { font: 700 32px var(--f-titulo); color: var(--dorado-osc); width: 30px; }
.memo__resp--no { color: var(--rojo); }
.memo__nota { font-size: 18px; color: var(--tenue); }
</style>
