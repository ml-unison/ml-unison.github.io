<script setup lang="ts">
// ¿Qué casa se parece más a A? La respuesta cambia al estandarizar.
import datos from '../datos/u01.json'
const C = (datos as any).casas
const casas = { A: [1500, 5], B: [1550, 9], C: [1800, 5] } as Record<string, number[]>
const d = (p: number[], q: number[], s = [1, 1]) => Math.hypot((p[0] - q[0]) / s[0], (p[1] - q[1]) / s[1])
const sd = [C.area_sd, C.calidad_sd]
const fmt = (v: number) => v.toFixed(v < 10 ? 2 : 0)
</script>

<template>
  <div class="dist">
    <div>
      <TablaMini :columnas="['casa', 'área (pies²)', 'calidad (1–10)']" :filas="Object.entries(casas).map(([k, v]) => [k, v[0], v[1]])" :tam="22" />
      <p class="dist__nota">Casas hipotéticas. Desviaciones estándar en los datos de Ames: área {{ C.area_sd }} pies², calidad {{ C.calidad_sd }}.</p>
    </div>
    <div class="dist__res">
      <div class="dist__fila">
        <div class="dist__titulo">Distancia con los datos crudos</div>
        <div><b>d(A, B)</b> = {{ fmt(d(casas.A, casas.B)) }} · <b>d(A, C)</b> = {{ fmt(d(casas.A, casas.C)) }}</div>
        <div class="dist__veredicto">B «se parece más» a A: la calidad casi no cuenta.</div>
      </div>
      <div v-click class="dist__fila">
        <div class="dist__titulo">Distancia estandarizada</div>
        <div><b>d(A, B)</b> = {{ fmt(d(casas.A, casas.B, sd)) }} · <b>d(A, C)</b> = {{ fmt(d(casas.A, casas.C, sd)) }}</div>
        <div class="dist__veredicto">Ahora C es la más parecida: ambas variables pesan según su variación.</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dist { display: grid; grid-template-columns: 1fr 1fr; gap: 56px; align-items: start; }
.dist__nota { font-size: 17px; color: var(--suave); margin-top: 12px; }
.dist__res { display: flex; flex-direction: column; gap: 26px; }
.dist__titulo { font: 650 14px var(--f-titulo); letter-spacing: .1em; text-transform: uppercase; color: var(--azul); margin-bottom: 6px; }
.dist__fila { font-size: 24px; }
.dist__fila b { font-weight: 650; color: var(--marino); }
.dist__veredicto { font-size: 20px; color: var(--tenue); margin-top: 4px; }
</style>
