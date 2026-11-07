<script setup lang="ts">
// Un bloque residual dibujado: la entrada a pasa por dos convoluciones (la corrección F) y además
// salta por encima de ellas («atajo» o identidad); al final se suman. Con `simple` se dibuja sin el atajo.
const props = withDefaults(defineProps<{ simple?: boolean }>(), { simple: false })
const W = 820, H = 320, cy = 190
// láminas: mapas de activación de 64 canales
const lam = (x: number, n = 5, lado = 90) =>
  Array.from({ length: n }, (_, k) => ({ x: x + k * 6, y: cy - lado / 2 - k * 6 + 12, lado }))
const etapas = [
  { x: 30, et: '𝐚[l]', color: '#24398A' },
  { x: 230, et: 'conv, ReLU', color: '#8A91A0' },
  { x: 420, et: 'conv', color: '#8A91A0' },
  { x: 680, et: '𝐚[l+1]', color: '#24398A' },
]
</script>

<template>
  <svg class="grafica" :viewBox="`0 0 ${W} ${H}`">
    <defs>
      <marker id="flecha-res" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#1A2133" />
      </marker>
    </defs>
    <g v-for="(e, j) in etapas" :key="j">
      <rect v-for="(l, k) in lam(e.x)" :key="k" :x="l.x" :y="l.y" :width="l.lado" :height="l.lado"
            fill="#FFFFFF" :stroke="e.color" stroke-width="1.5" />
      <text :x="e.x + 57" :y="cy + 80" text-anchor="middle" style="fill: #1A2133; font-weight: 700; font-size: 24px">{{ e.et }}</text>
    </g>
    <!-- flujo principal -->
    <line x1="160" :y1="cy" x2="222" :y2="cy" stroke="#1A2133" stroke-width="2" marker-end="url(#flecha-res)" />
    <line x1="350" :y1="cy" x2="412" :y2="cy" stroke="#1A2133" stroke-width="2" marker-end="url(#flecha-res)" />
    <template v-if="!props.simple">
      <line x1="540" :y1="cy" x2="588" :y2="cy" stroke="#1A2133" stroke-width="2" marker-end="url(#flecha-res)" />
      <circle cx="610" :cy="cy" r="20" fill="#FBF6E6" stroke="#C9971C" stroke-width="2.5" />
      <text x="610" :y="cy + 9" text-anchor="middle" style="fill: #A87C12; font-weight: 800; font-size: 28px">+</text>
      <line x1="632" :y1="cy" x2="672" :y2="cy" stroke="#1A2133" stroke-width="2" marker-end="url(#flecha-res)" />
      <!-- atajo -->
      <path :d="`M 95 ${cy - 60} C 95 10, 610 10, 610 ${cy - 24}`" fill="none" stroke="#C9971C" stroke-width="3.5" marker-end="url(#flecha-res)" />
      <text x="352" y="36" text-anchor="middle" style="fill: #A87C12; font-weight: 700; font-size: 26px">atajo: 𝐚[l] pasa tal cual</text>
      <text x="390" :y="cy + 112" text-anchor="middle" style="fill: #5E6678; font-size: 22px">la corrección F[l](𝐚[l])</text>
      <path :d="`M 230 ${cy + 92} L 540 ${cy + 92}`" stroke="#8A91A0" stroke-width="1.5" />
    </template>
    <line v-else x1="540" :y1="cy" x2="672" :y2="cy" stroke="#1A2133" stroke-width="2" marker-end="url(#flecha-res)" />
  </svg>
</template>
