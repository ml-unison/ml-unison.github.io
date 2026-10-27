<script setup lang="ts">
// Puntajes del bosque aleatorio en los 7,500 clientes de prueba: histogramas por clase, un umbral móvil,
// la matriz de confusión que resulta y el punto correspondiente en la curva ROC.
import { computed, ref } from 'vue'
import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import datos from '../datos/u03.json'
const C = (datos as any).credito
const pos: number[] = C.puntajes_pos, neg: number[] = C.puntajes_neg
const t = ref(0.5)
// cuántos puntajes ordenados son ≥ t (búsqueda binaria)
const mayores = (a: number[], v: number) => { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m } return a.length - lo }
const M = computed(() => { const VP = mayores(pos, t.value), FP = mayores(neg, t.value); return { VP, FP, FN: pos.length - VP, VN: neg.length - FP } })
const met = computed(() => { const { VP, FP, FN, VN } = M.value, n = VP + FP + FN + VN
  return { exactitud: (VP + VN) / n, precision: VP + FP ? VP / (VP + FP) : NaN, exhaustividad: VP / (VP + FN), tfp: FP / (FP + VN) } })
const W = 640, H = 300, m = { l: 20, r: 20, t: 20, b: 44 }
const x = scaleLinear().domain([0, 1]).range([m.l, W - m.r])
const bw = (W - m.l - m.r) / 40
const maxh = Math.max(...C.hist_neg, ...C.hist_pos)
const y = scaleLinear().domain([0, maxh]).range([0, (H - m.t - m.b) / 2 - 6])
const mid = m.t + (H - m.t - m.b) / 2
const R = (datos as any).credito.roc.bosque
const S = 230, ms = 34
const rx = scaleLinear().domain([0, 1]).range([ms, S - 8]), ry = scaleLinear().domain([0, 1]).range([S - ms, 8])
const roc = line<number>().x((_, i) => rx(R.fpr[i])).y((v) => ry(v))(R.tpr)!
const pct = (v: number) => Number.isNaN(v) ? '—' : `${(100 * v).toFixed(1)} %`
</script>

<template>
  <div class="ui">
    <div>
      <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
        <rect v-for="(c, i) in C.hist_neg" :key="`n${i}`" :x="x(i / 40) + 1" :y="mid - y(c)" :width="bw - 2" :height="y(c)" fill="#24398A" :opacity="(i + 0.5) / 40 >= t ? 1 : 0.35" />
        <rect v-for="(c, i) in C.hist_pos" :key="`p${i}`" :x="x(i / 40) + 1" :y="mid + 2" :width="bw - 2" :height="y(c)" fill="#D5433C" :opacity="(i + 0.5) / 40 >= t ? 1 : 0.35" />
        <line :x1="m.l" :x2="W - m.r" :y1="mid + 1" :y2="mid + 1" stroke="#8A91A0" />
        <line :x1="x(t)" :x2="x(t)" :y1="m.t - 8" :y2="H - m.b + 6" stroke="#1A2133" stroke-width="2.5" />
        <text :x="W - m.r" :y="m.t + 14" text-anchor="end" style="fill: #1A2133; font-weight: 600">sí pagaron (5,841)</text>
        <text :x="W - m.r" :y="H - m.b - 6" text-anchor="end" style="fill: #1A2133; font-weight: 600">incumplieron (1,659)</text>
        <g class="eje"><text v-for="v in [0, 0.25, 0.5, 0.75, 1]" :key="v" :x="x(v)" :y="H - 14" text-anchor="middle">{{ v }}</text></g>
      </svg>
      <div class="deslizador"><span>Umbral t</span><input v-model.number="t" type="range" min="0.02" max="0.9" step="0.01" /><b>{{ t.toFixed(2) }}</b></div>
      <p class="ui__nota">Puntaje del bosque aleatorio (probabilidad estimada de incumplir). A la derecha del umbral, el modelo dice «incumple».</p>
    </div>
    <div class="ui__lado">
      <table class="ui__mc">
        <tr><th></th><th>dice «paga»</th><th>dice «incumple»</th></tr>
        <tr><th>pagó</th><td>VN {{ M.VN.toLocaleString('es-MX') }}</td><td class="mal">FP {{ M.FP.toLocaleString('es-MX') }}</td></tr>
        <tr><th>incumplió</th><td class="mal">FN {{ M.FN.toLocaleString('es-MX') }}</td><td>VP {{ M.VP.toLocaleString('es-MX') }}</td></tr>
      </table>
      <div class="ui__met">
        <div>exactitud <b>{{ pct(met.exactitud) }}</b></div><div>precisión <b>{{ pct(met.precision) }}</b></div>
        <div>exhaustividad (TVP) <b>{{ pct(met.exhaustividad) }}</b></div><div>tasa de falsos positivos <b>{{ pct(met.tfp) }}</b></div>
      </div>
      <svg class="grafica" :viewBox="`0 0 ${S} ${S}`" style="width: 230px">
        <line :x1="rx(0)" :y1="ry(0)" :x2="rx(1)" :y2="ry(1)" stroke="#C5C9D3" stroke-dasharray="4 4" />
        <path :d="roc" fill="none" stroke="#24398A" stroke-width="2.5" />
        <circle :cx="rx(met.tfp)" :cy="ry(met.exhaustividad)" r="7" fill="#C9971C" stroke="#fff" stroke-width="2" />
        <g class="eje"><line :x1="ms" :x2="S - 8" :y1="S - ms" :y2="S - ms" /><line :x1="ms" :x2="ms" :y1="8" :y2="S - ms" />
          <text :x="(ms + S) / 2" :y="S - 8" text-anchor="middle" style="font-size: 14px">TFP</text>
          <text :x="12" :y="(S - ms) / 2" text-anchor="middle" style="font-size: 14px" :transform="`rotate(-90 12 ${(S - ms) / 2})`">TVP</text></g>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.ui { display: grid; grid-template-columns: 640px 1fr; gap: 36px; }
.ui .deslizador input[type=range] { flex: 1 1 auto; }
.ui__nota { font-size: 16px; color: var(--tenue); }
.ui__lado { display: flex; flex-direction: column; gap: 12px; }
.ui__mc { border-collapse: collapse; font-size: 18px; }
.ui__mc th { font-weight: 600; color: var(--tenue); padding: 4px 10px; font-size: 16px; text-align: left; }
.ui__mc td { padding: 8px 12px; border: 1px solid var(--linea); font-variant-numeric: tabular-nums; }
.ui__mc td.mal { background: var(--tinte-rojo); }
.ui__met { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 18px; font-size: 18px; }
.ui__met b { color: var(--azul); }
</style>
