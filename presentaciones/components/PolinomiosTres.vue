<script setup lang="ts">
import datos from '../datos/u01.json'

const P = datos.polinomios
const fmt = (v: number) => {
  if (v < 100) return v.toFixed(2)
  const [m, e] = v.toExponential(0).split('e+')
  return `${m} × 10<sup>${e}</sup>`
}
const paneles = [
  { g: 1, diag: 'subajuste', clase: 'mal' },
  { g: 4, diag: 'buen compromiso', clase: 'bien' },
  { g: 12, diag: 'sobreajuste', clase: 'mal' },
]
</script>

<template>
  <div class="tres">
    <div v-for="(p, k) in paneles" :key="p.g" class="tres__panel">
      <div class="tres__cab">
        <span class="tres__grado">Grado {{ p.g }}</span>
        <span class="tres__diag" :class="p.clase">{{ p.diag }}</span>
      </div>
      <PanelPolinomio :grado="p.g" :eje-y="k === 0" :ancho="k === 0 ? 392 : 360" />
      <div class="tres__err">
        <span>E<sub>in</sub> = <b v-html="fmt(P.ein[p.g])" /></span>
        <span>E<sub>out</sub> = <b :class="{ alto: P.eout[p.g] > 1 }" v-html="fmt(P.eout[p.g])" /></span>
      </div>
    </div>
  </div>
  <div class="tres__leyenda">
    <span><i class="sw-punto" /> datos <b>𝒟</b> (n = 15)</span>
    <span><i class="sw-linea gris" /> función desconocida f</span>
    <span><i class="sw-linea azul" /> hipótesis ajustada g</span>
  </div>
</template>

<style scoped>
.tres { display: grid; grid-template-columns: 392fr 360fr 360fr; gap: 26px; }
.tres__cab { display: flex; align-items: baseline; justify-content: space-between; padding: 0 6px 6px 46px; }
.tres__panel:not(:first-child) .tres__cab, .tres__panel:not(:first-child) .tres__err { padding-left: 14px; }
.tres__grado { font-family: var(--f-titulo); font-weight: 650; font-size: 22px; color: var(--marino); }
.tres__diag { font-family: var(--f-titulo); font-weight: 600; font-size: 13px; letter-spacing: .07em; text-transform: uppercase; padding: 4px 10px; border-radius: 20px; }
.tres__diag.mal { background: var(--tinte-rojo); color: #B23028; }
.tres__diag.bien { background: #EEF6EA; color: #3F7F2E; }
.tres__err { display: flex; gap: 22px; padding: 2px 0 0 46px; font-size: 20px; color: var(--tenue); }
.tres__err b { color: var(--tinta); font-weight: 650; font-variant-numeric: tabular-nums; }
.tres__err b.alto { color: #B23028; }
.tres__leyenda { display: flex; gap: 34px; justify-content: center; font-size: 19px; color: var(--tenue); margin-top: 16px; }
.tres__leyenda span { display: inline-flex; align-items: center; gap: 10px; }
.sw-punto { width: 14px; height: 14px; border-radius: 50%; background: var(--dorado); display: inline-block; }
.sw-linea { width: 28px; height: 0; border-top: 3.5px solid var(--azul); display: inline-block; }
.sw-linea.gris { border-top: 2.5px dotted #8A91A0; }
</style>
