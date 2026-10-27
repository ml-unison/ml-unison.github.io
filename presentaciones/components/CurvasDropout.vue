<script setup lang="ts">
// Con y sin dropout, para cada optimizador: pérdida de entrenamiento (punteada) y de validación (continua).
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u06-02.json'
const C = (datos as any).corridas as any[]
const W = 370, H = 360, m = { l: 52, r: 14, t: 40, b: 60 }
const x = scaleLinear().domain([1, 30]).range([m.l, W - m.r])
const y = scaleLinear().domain([0, 1.0]).range([H - m.b, m.t]).clamp(true)
const d = (v: number[]) => line<number>().x((_, i) => x(i + 1)).y(w => y(w))(v)!
const busca = (n: string) => C.find(c => c.nombre === n)
const paneles = ['SGD', 'SGD + momento', 'Adam'].map(n => ({ n, sin: busca(n), con: busca(n + ' + dropout') }))
</script>

<template>
  <div>
    <div class="cd__leyenda">
      <span><i style="background: #D5433C" /> sin dropout</span>
      <span><i style="background: #24398A" /> con dropout (p = 0.5)</span>
      <span><i class="punteada" /> entrenamiento</span>
      <span><i style="background: #5E6678" /> validación</span>
    </div>
    <div class="cd">
      <svg v-for="p in paneles" :key="p.n" class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <text :x="m.l" :y="20" style="fill: #20305A; font-weight: 650; font-size: 20px">{{ p.n }}</text>
        <Ejes :x="x" :y="y" :x-ticks="[1, 10, 20, 30]" :y-ticks="[0, 0.2, 0.4, 0.6, 0.8, 1.0]" x-titulo="época" />
        <path :d="d(p.sin.pe)" fill="none" stroke="#D5433C" stroke-width="2.4" stroke-dasharray="6 5" />
        <path :d="d(p.con.pe)" fill="none" stroke="#24398A" stroke-width="2.4" stroke-dasharray="6 5" />
        <path :d="d(p.sin.pv)" fill="none" stroke="#D5433C" stroke-width="3.2" />
        <path :d="d(p.con.pv)" fill="none" stroke="#24398A" stroke-width="3.2" />
        <template v-if="p.n === 'Adam'">
          <text :x="x(30)" :y="y(p.sin.pv_fin) - 12" text-anchor="end" style="fill: #1A2133; font-weight: 600; font-size: 17px">{{ p.sin.pv_fin.toFixed(2) }}</text>
          <text :x="x(30)" :y="y(p.con.pv_fin) - 12" text-anchor="end" style="fill: #1A2133; font-weight: 600; font-size: 17px">{{ p.con.pv_fin.toFixed(2) }}</text>
        </template>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.cd { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
.cd__leyenda { display: flex; gap: 30px; justify-content: center; font-size: 19px; color: var(--tinta); margin-bottom: 4px; }
.cd__leyenda i { display: inline-block; width: 26px; height: 4px; vertical-align: 5px; margin-right: 6px; border-radius: 2px; }
.cd__leyenda i.punteada { background: repeating-linear-gradient(90deg, #5E6678 0 6px, transparent 6px 10px); }
</style>
