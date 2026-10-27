<script setup lang="ts">
// Coeficientes de ridge y lasso en los datos de diabetes al variar λ.
import datos from '../datos/u03.json'
const C = (datos as any).caminos
const props = defineProps<{ metodo: 'ridge' | 'lasso' }>()
// λ de mayor a menor: de izquierda a derecha los coeficientes crecen hasta los de mínimos cuadrados
const lam = props.metodo === 'ridge' ? [...C.lam_ridge].reverse() : C.lam_lasso
const coefs = props.metodo === 'ridge' ? C.ridge.map((v: number[]) => [...v].reverse()) : C.lasso
// solo las 4 variables de mayor coeficiente llevan etiqueta y color; el resto en gris
const orden = coefs.map((v: number[], i: number) => [Math.max(...v.map(Math.abs)), i]).sort((a: number[], b: number[]) => b[0] - a[0]).map((p: number[]) => p[1])
const top = orden.slice(0, 4)
const series: Record<string, number[]> = {}
const colores: string[] = []
const fijos = ['#24398A', '#D5433C', '#C9971C', '#6B7385']
top.forEach((i: number, k: number) => { series[C.nombres[i]] = coefs[i]; colores.push(fijos[k]) })
orden.slice(4).forEach((i: number) => { series[' '.repeat(i + 1)] = coefs[i]; colores.push('#C5C9D3') })
</script>

<template>
  <LineasLog :x="lam" :series="series" x-log :y-log="false" :y-dominio="metodo === 'ridge' ? [-45, 45] : [-45, 45]" :y-ticks="[-40, -20, 0, 20, 40]"
             :x-dominio="[Math.max(...lam), Math.min(...lam)]" :x-ticks="metodo === 'ridge' ? [0.01, 1, 100, 10000] : [0.1, 1, 10]"
             :x-titulo="`λ, de mayor a menor (escala log)`" y-titulo="coeficiente" :colores="colores" :ancho="560" :alto="380" :derecha="110" />
</template>
