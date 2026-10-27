<template>
  <div class="ft">
    <div class="ft__fila">
      <code class="ft__codigo">escalador.fit(X_entrenamiento)</code>
      <span class="ft__flecha">⟶</span>
      <div class="ft__aprende">aprende <b>θ</b>: <span>medias, desviaciones, medianas, categorías…</span></div>
    </div>
    <div v-click class="ft__fila">
      <code class="ft__codigo">escalador.transform(X_entrenamiento)</code>
      <span class="ft__flecha">⟶</span>
      <div class="ft__aprende">aplica <b>θ</b> a los datos con los que aprendió</div>
    </div>
    <div v-click class="ft__fila">
      <code class="ft__codigo ft__codigo--oro">escalador.transform(X_prueba)</code>
      <span class="ft__flecha">⟶</span>
      <div class="ft__aprende">aplica <b>la misma θ</b>, sin volver a aprender nada</div>
    </div>
  </div>
</template>

<style scoped>
.ft { display: flex; flex-direction: column; gap: 28px; }
.ft__fila { display: grid; grid-template-columns: 500px 60px 1fr; align-items: center; }
.ft__codigo { font-family: var(--f-codigo); font-size: 22px; background: var(--fondo-alt); border: 1px solid var(--linea); border-radius: 8px; padding: 12px 16px; color: var(--marino); }
.ft__codigo--oro { background: var(--tinte-dorado); border-color: #EBD9A0; }
.ft__flecha { font-size: 30px; color: var(--azul-claro); text-align: center; }
.ft__aprende { font-size: 24px; }
.ft__aprende b { color: var(--azul); font-family: var(--f-titulo); }
.ft__aprende span { color: var(--tenue); }
</style>
