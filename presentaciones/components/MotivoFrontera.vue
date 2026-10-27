<script setup lang="ts">
// Motivo decorativo: dos clases y una frontera de decisión no lineal.
// Los puntos se generan con una semilla fija, así la figura siempre es la misma.
const props = withDefaults(defineProps<{ ancho?: number, alto?: number, n?: number, semilla?: number }>(),
  { ancho: 534, alto: 720, n: 150, semilla: 7 })

function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const rnd = mulberry32(props.semilla)
const W = props.ancho, H = props.alto
const frontera = (x: number) => H * (0.52 + 0.16 * Math.sin((x / W) * 2 * Math.PI * 0.9 + 0.6) + 0.05 * Math.sin((x / W) * 9))

const puntos = Array.from({ length: props.n }, () => {
  const x = rnd() * W, y = rnd() * H
  const d = y - frontera(x)
  // cerca de la frontera algunos puntos quedan "del otro lado": ruido de etiqueta
  const clase = (d + (rnd() - 0.5) * 70) > 0 ? 1 : 0
  return { x, y, clase, r: 4 + rnd() * 4, o: 0.55 + rnd() * 0.45 }
})
const curva = Array.from({ length: 121 }, (_, i) => {
  const x = (i / 120) * W
  return `${i ? 'L' : 'M'}${x.toFixed(1)},${frontera(x).toFixed(1)}`
}).join('')
const region = `${curva}L${W},${H}L0,${H}Z`
</script>

<template>
  <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid slice" width="100%" height="100%" aria-hidden="true">
    <path :d="region" fill="#24398A" opacity="0.55" />
    <path :d="curva" fill="none" stroke="#E1B73D" stroke-width="2.5" stroke-dasharray="10 8" />
    <circle v-for="(p, i) in puntos" :key="i" :cx="p.x" :cy="p.y" :r="p.r"
            :fill="p.clase ? '#E1B73D' : 'none'" :stroke="p.clase ? 'none' : '#8DA3C9'"
            :stroke-width="2.2" :opacity="p.o" />
  </svg>
</template>
