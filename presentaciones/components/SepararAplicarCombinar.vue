<script setup lang="ts">
// groupby en tres pasos: separar por grupo, aplicar una función, combinar los resultados.
import datos from '../datos/u01.json'
import { RIO } from './rios'

const M = (datos as any).presas.muestra.filter((f: any) => RIO[f.presa] !== 'Otros')
  .map((f: any) => ({ ...f, rio: RIO[f.presa] }))
  .sort((a: any, b: any) => a.rio.localeCompare(b.rio))
const rios = ['Mayo', 'Sonora', 'Yaqui']
const grupo = (r: string) => M.filter((f: any) => f.rio === r)
const media = (r: string) => Math.round(grupo(r).reduce((s: number, f: any) => s + f.llenado, 0) / grupo(r).length * 10) / 10
</script>

<template>
  <div class="sac">
    <div>
      <div class="sac__paso">la tabla</div>
      <TablaMini :columnas="['presa', 'río', 'llenado']" :filas="M.map((f: any) => [f.presa, f.rio, f.llenado])" :tam="19" />
    </div>
    <div v-click class="sac__col">
      <div class="sac__paso">1 · separar por río</div>
      <div class="sac__grupos">
        <TablaMini v-for="r in rios" :key="r" :columnas="['presa', 'llenado']" :filas="grupo(r).map((f: any) => [f.presa, f.llenado])" :grupos="grupo(r).map(() => r)" :tam="19" />
      </div>
    </div>
    <div v-click class="sac__col">
      <div class="sac__paso">2 · aplicar <code>mean</code> · 3 · combinar</div>
      <TablaMini :columnas="['río', 'llenado medio']" :filas="rios.map(r => [r, media(r)])" :grupos="rios" :tam="19" />
    </div>
  </div>
</template>

<style scoped>
.sac { display: grid; grid-template-columns: auto auto auto; gap: 48px; align-items: start; justify-content: center; }
.sac__paso { font: 650 14px var(--f-titulo); letter-spacing: .1em; text-transform: uppercase; color: var(--suave); margin-bottom: 10px; }
.sac__paso code { text-transform: none; letter-spacing: 0; font-size: 13px; }
.sac__grupos { display: flex; flex-direction: column; gap: 12px; }
</style>
