<script setup lang="ts">
// El ColumnTransformer de la libreta u01-e02: cada grupo de columnas con su tratamiento.
const grupos = [
  { cols: 'GrLivArea, LotArea', pasos: ['log1p', 'StandardScaler'], color: '#24398A' },
  { cols: 'LotFrontage, OverallQual, …', pasos: ['SimpleImputer(median, indicador)', 'StandardScaler'], color: '#496EA5' },
  { cols: 'YearBuilt, YearRemodAdd, YrSold', pasos: ['edades()', 'StandardScaler'], color: '#8DA3C9' },
  { cols: 'Neighborhood, BldgType, CentralAir', pasos: ['OneHotEncoder'], color: '#C9971C' },
  { cols: 'KitchenQual, ExterQual, PoolQC, …', pasos: ['«Ninguna»', 'OrdinalEncoder', 'StandardScaler'], color: '#D5433C' },
]
const H = 400, fila = 72, y0 = 34
const yc = (i: number) => y0 + i * fila + 18
const xm = 880, ym = H / 2
const curva = (i: number) => `M818,${yc(i)} C848,${yc(i)} 848,${ym - 112 + i * 56} ${xm},${ym - 112 + i * 56}`
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 1120 ${H}`" role="img" aria-label="ColumnTransformer con cinco grupos de columnas">
    <text x="0" y="16" class="cab">columnas</text>
    <text x="300" y="16" class="cab">transformación (aprende con entrenamiento)</text>
    <g v-for="(g, i) in grupos" :key="i">
      <rect x="0" :y="yc(i) - 16" width="6" height="32" rx="3" :fill="g.color" />
      <text x="18" :y="yc(i) + 6" class="cols">{{ g.cols }}</text>
      <text x="300" :y="yc(i) + 6" class="paso">
        <tspan v-for="(p, k) in g.pasos" :key="k">{{ k ? '  →  ' : '' }}{{ p }}</tspan>
      </text>
      <path :d="curva(i)" fill="none" :stroke="g.color" stroke-width="2.5" />
    </g>
    <rect :x="xm" :y="ym - 140" width="28" height="280" rx="4" fill="#F6F7FA" stroke="#20305A" stroke-width="2" />
    <rect v-for="(g, i) in grupos" :key="`m${i}`" :x="xm + 4" :y="ym - 136 + i * 55" width="20" height="52" rx="2" :fill="g.color" opacity="0.85" />
    <text :x="xm + 14" :y="ym + 168" text-anchor="middle" class="cab">matriz X</text>
    <line :x1="xm + 36" :x2="xm + 76" :y1="ym" :y2="ym" stroke="#20305A" stroke-width="2.5" />
    <path :d="`M${xm + 70},${ym - 6} L${xm + 80},${ym} L${xm + 70},${ym + 6}`" fill="none" stroke="#20305A" stroke-width="2.5" />
    <text :x="xm + 90" :y="ym - 4" class="modelo">RidgeCV</text>
    <text :x="xm + 90" :y="ym + 22" class="cols">sobre log(precio)</text>
  </svg>
</template>

<style scoped>
.cab { font: 650 13px var(--f-titulo); letter-spacing: .1em; text-transform: uppercase; fill: #8A91A0; }
.cols { font: 400 18px var(--f-texto); fill: #1A2133; }
.paso { font: 500 15px var(--f-codigo); fill: #20305A; }
.modelo { font: 650 22px var(--f-titulo); fill: #20305A; }
</style>
