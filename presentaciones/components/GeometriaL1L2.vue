<script setup lang="ts">
// Curvas de nivel del error cuadrático (elipses) y la región permitida: círculo (ridge) o rombo (lasso).
// El punto de contacto se calcula: es el punto de la frontera con menor error.
const O = [150, 220], R = 80                     // origen de los ejes y «radio» C de la región
const c = [285, 95], a = 1.0, b = 0.45, th = (-30 * Math.PI) / 180
const Q = (p: number[]) => { const dx = p[0] - c[0], dy = p[1] - c[1]
  const u = dx * Math.cos(th) + dy * Math.sin(th), v = -dx * Math.sin(th) + dy * Math.cos(th)
  return (u / a) ** 2 + (v / b) ** 2 }
const frontera = (tipo: string) => Array.from({ length: 1440 }, (_, i) => {
  const t = (i / 1440) * 2 * Math.PI
  if (tipo === 'circulo') return [O[0] + R * Math.cos(t), O[1] + R * Math.sin(t)]
  const ct = Math.cos(t), st = Math.sin(t), k = R / (Math.abs(ct) + Math.abs(st))
  return [O[0] + k * ct, O[1] + k * st]
})
const toque = (tipo: string) => frontera(tipo).reduce((m, p) => Q(p) < Q(m) ? p : m)
const elipses = (tipo: string) => { const L = Q(toque(tipo)); return [0.25, 0.55, 1].map(f => ({ rx: a * Math.sqrt(L * f), ry: b * Math.sqrt(L * f) })) }
const panels = [{ t: 'Ridge: ‖w‖₂ ≤ C', tipo: 'circulo' }, { t: 'Lasso: ‖w‖₁ ≤ C', tipo: 'rombo' }]
const deg = (th * 180) / Math.PI
</script>

<template>
  <div class="gl">
    <div v-for="p in panels" :key="p.t">
      <svg class="grafica" viewBox="0 0 400 330">
        <line x1="20" x2="390" :y1="O[1]" :y2="O[1]" stroke="#8A91A0" stroke-width="1.5" />
        <line :x1="O[0]" :x2="O[0]" y1="10" y2="325" stroke="#8A91A0" stroke-width="1.5" />
        <text x="378" :y="O[1] + 24" style="font-size: 17px">w₁</text><text :x="O[0] + 8" y="26" style="font-size: 17px">w₂</text>
        <circle v-if="p.tipo === 'circulo'" :cx="O[0]" :cy="O[1]" :r="R" fill="#C0CEE3" opacity="0.7" stroke="#24398A" stroke-width="2.5" />
        <polygon v-else :points="`${O[0]},${O[1] - R} ${O[0] + R},${O[1]} ${O[0]},${O[1] + R} ${O[0] - R},${O[1]}`" fill="#C0CEE3" opacity="0.7" stroke="#24398A" stroke-width="2.5" />
        <ellipse v-for="(e, k) in elipses(p.tipo)" :key="k" :cx="c[0]" :cy="c[1]" :rx="e.rx" :ry="e.ry" :transform="`rotate(${deg} ${c[0]} ${c[1]})`" fill="none" stroke="#D5433C" stroke-width="1.8" />
        <circle :cx="c[0]" :cy="c[1]" r="5" fill="#D5433C" />
        <text :x="c[0] - 4" :y="c[1] - 14" text-anchor="middle" style="font-size: 15px">mínimos cuadrados</text>
        <circle :cx="toque(p.tipo)[0]" :cy="toque(p.tipo)[1]" r="8" fill="#C9971C" stroke="#fff" stroke-width="2" />
      </svg>
      <div class="gl__t">{{ p.t }}</div>
    </div>
  </div>
</template>

<style scoped>
.gl { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
.gl__t { text-align: center; font-weight: 600; font-size: 20px; color: var(--marino); }
</style>
