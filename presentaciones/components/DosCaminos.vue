<script setup lang="ts">
// Programación tradicional vs. aprendizaje: la misma ecuación con los términos intercambiados.
import datos from '../datos/u01.json'
const D = datos.digitos as any
const ejemplos = [D.sietes[0], D.unos[0], D.sietes[1], D.mosaico[0][4], D.unos[1]]
const etiquetas = ['7', '1', '7', '4', '1']
</script>

<template>
  <div class="caminos">
    <div class="camino">
      <div class="camino__nombre">Programación tradicional</div>
      <div class="camino__ecuacion">
        <span class="t t--regla">reglas</span><span class="op">+</span><span class="t">datos</span>
        <svg class="flecha-larga" viewBox="0 0 120 20"><line x1="2" y1="10" x2="112" y2="10" /><path d="M104,3 L116,10 L104,17" /></svg>
        <span class="t t--resp">respuestas</span>
      </div>
      <div class="camino__nota">Alguien escribe las reglas; la computadora las aplica.</div>
    </div>
    <div v-click class="camino">
      <div class="camino__nombre">Aprendizaje automático</div>
      <div class="camino__ecuacion">
        <span class="t">datos</span><span class="op">+</span><span class="t t--resp">respuestas</span>
        <svg class="flecha-larga" viewBox="0 0 120 20"><line x1="2" y1="10" x2="112" y2="10" /><path d="M104,3 L116,10 L104,17" /></svg>
        <span class="t t--regla">reglas</span>
      </div>
      <div class="camino__ejemplos">
        <div v-for="(d, i) in ejemplos" :key="i" class="par">
          <Digito :px="d" :tam="44" />
          <span>{{ etiquetas[i] }}</span>
        </div>
        <span class="tenue">…miles de ejemplos etiquetados</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.caminos { display: flex; flex-direction: column; gap: 46px; margin-top: 10px; }
.camino__nombre { font-family: var(--f-titulo); font-weight: 650; font-size: 15px; letter-spacing: .12em; text-transform: uppercase; color: var(--suave); margin-bottom: 10px; }
.camino__ecuacion { display: flex; align-items: center; gap: 22px; font-family: var(--f-titulo); font-weight: 650; font-size: 46px; letter-spacing: -0.02em; color: var(--marino); }
.op { color: var(--azul-claro); font-weight: 400; }
.t--regla { color: var(--azul); }
.t--resp { color: var(--dorado-osc); }
.flecha-larga { width: 120px; height: 20px; fill: none; stroke: var(--azul-claro); stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }
.camino__nota { font-size: 22px; color: var(--tenue); margin-top: 8px; }
.camino__ejemplos { display: flex; align-items: center; gap: 18px; margin-top: 18px; }
.par { display: flex; align-items: center; gap: 8px; background: var(--fondo-alt); border-radius: 10px; padding: 6px 12px 6px 6px; }
.par span { font-family: var(--f-titulo); font-weight: 700; font-size: 24px; color: var(--dorado-osc); }
.tenue { font-size: 20px; color: var(--suave); }
</style>
