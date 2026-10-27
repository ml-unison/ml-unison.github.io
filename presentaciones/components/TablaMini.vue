<script setup lang="ts">
// Tabla pequeña para explicar operaciones con datos: resalta columnas, filas o grupos.
const props = withDefaults(defineProps<{
  columnas: string[], filas: (string | number)[][],
  marcarCols?: number[], marcarFilas?: number[], atenuarFilas?: number[], nuevaCol?: number,
  grupos?: string[], tam?: number, clave?: number, malas?: number[],
}>(), { marcarCols: () => [], marcarFilas: () => [], atenuarFilas: () => [], grupos: () => [], tam: 18, malas: () => [] })

const COLORES: Record<string, string> = { Yaqui: '#EFF2F9', Mayo: '#FBF6E6', Sonora: '#FCEFEE', Otros: '#F6F7FA' }
const BORDES: Record<string, string> = { Yaqui: '#24398A', Mayo: '#C9971C', Sonora: '#D5433C', Otros: '#8A91A0' }
const fondo = (i: number, j: number) => {
  if (props.malas.includes(i)) return '#FCEFEE'
  if (props.grupos[i]) return COLORES[props.grupos[i]]
  if (props.nuevaCol === j) return '#FBF6E6'
  if (props.marcarCols.includes(j) || props.marcarFilas.includes(i)) return '#EFF2F9'
  return 'transparent'
}
const num = (v: string | number) => typeof v === 'number'
</script>

<template>
  <table class="tmini" :style="{ fontSize: `${tam}px` }">
    <thead>
      <tr>
        <th v-for="(c, j) in columnas" :key="j" :class="{ marcada: marcarCols.includes(j) || clave === j, nueva: nuevaCol === j }">{{ c }}</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="(f, i) in filas" :key="i" :class="{ atenuada: atenuarFilas.includes(i) }"
          :style="grupos[i] ? { boxShadow: `inset 4px 0 0 ${BORDES[grupos[i]]}` } : {}">
        <td v-for="(v, j) in f" :key="j" :class="{ num: num(v), clave: clave === j }" :style="{ background: fondo(i, j) }">{{ num(v) ? (v as number).toLocaleString('es-MX') : v }}</td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
.tmini { border-collapse: collapse; width: auto; font-family: var(--f-texto); }
:root .slidev-layout .tmini th { font-family: var(--f-codigo); font-size: 0.78em; font-weight: 600; letter-spacing: 0; text-transform: none; color: var(--tenue); padding: 6px 14px 8px; border-bottom: 2px solid var(--marino); text-align: left; }
.tmini th.marcada { color: var(--azul); }
.tmini th.nueva { color: var(--dorado-osc); }
:root .slidev-layout .tmini td { padding: 6px 14px; border-bottom: 1px solid var(--linea); font-weight: 400; color: var(--tinta); white-space: nowrap; }
:root .slidev-layout .tmini td:first-child { font-weight: 400; color: var(--tinta); }
.tmini td.num { text-align: right; font-variant-numeric: tabular-nums; }
.tmini td.clave { font-family: var(--f-codigo); font-size: 0.85em; color: var(--azul) !important; }
.tmini tr.atenuada td { opacity: 0.28; }
</style>
