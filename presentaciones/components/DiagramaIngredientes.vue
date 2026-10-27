<template>
  <div class="di">
    <div class="di__desconocido">
      <div class="bloque bloque--oro">Función objetivo<small><M t="f:\mathcal{X}\to\mathcal{Y}" /></small></div>
      <div class="bloque bloque--oro">Distribución<small><M t="P(\mathbf{x}, y)" /></small></div>
      <div class="di__nota">desconocidas</div>
    </div>
    <Flecha :largo="50" />
    <div class="bloque di__datos">Datos de entrenamiento<small><M t="\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^{n}" /></small></div>
    <Flecha :largo="50" />
    <div class="di__alg">
      <div class="bloque bloque--clave">Algoritmo<small><M t="\mathcal{A}" /></small></div>
      <svg class="di__arriba" viewBox="0 0 16 40" aria-hidden="true"><line x1="8" y1="38" x2="8" y2="8" stroke="#20305A" stroke-width="2.4" /><path d="M2,12 L8,2 L14,12" fill="none" stroke="#20305A" stroke-width="2.4" stroke-linejoin="round" /></svg>
      <div class="bloque">Espacio de hipótesis<small><M t="\mathcal{H}" /></small></div>
    </div>
    <Flecha :largo="50" />
    <div class="bloque bloque--resultado di__g">Hipótesis final<small><M t="g \approx f" /></small></div>
  </div>
</template>

<style scoped>
.di { display: flex; align-items: center; justify-content: space-between; }
.di__desconocido { display: flex; flex-direction: column; gap: 12px; padding: 14px; border: 2px dashed var(--dorado); border-radius: 14px; position: relative; }
.di__nota { position: absolute; top: -14px; left: 16px; background: #fff; padding: 0 8px; font-family: var(--f-titulo); font-size: 13px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; color: var(--dorado-osc); }
.di__datos { max-width: 250px; }
.di__alg { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.di__arriba { width: 16px; height: 40px; }
.di__g { padding: 22px 18px; }
</style>
