<template>
  <div class="la">
    <div>
      <div class="la__titulo">largo · una fila por observación</div>
      <TablaMini :columnas="['fecha', 'presa', 'llenado']" :tam="19" :filas="[
        ['2022-09-15', 'La Angostura', 89.2], ['2022-09-15', 'El Novillo', 99.0], ['2022-09-15', 'Oviachic', 36.9],
        ['2022-09-16', 'La Angostura', 89.5], ['2022-09-16', 'El Novillo', 99.1], ['2022-09-16', 'Oviachic', 37.3]]" />
    </div>
    <div class="la__ops">
      <div><code>pivot</code> →</div>
      <div>← <code>melt</code></div>
    </div>
    <div>
      <div class="la__titulo">ancho · una columna por serie</div>
      <TablaMini :columnas="['fecha', 'La Angostura', 'El Novillo', 'Oviachic']" :tam="19" :filas="[
        ['2022-09-15', 89.2, 99.0, 36.9], ['2022-09-16', 89.5, 99.1, 37.3]]" />
    </div>
  </div>
</template>

<style scoped>
.la { display: flex; align-items: center; gap: 40px; justify-content: center; }
.la__titulo { font: 650 14px var(--f-titulo); letter-spacing: .1em; text-transform: uppercase; color: var(--suave); margin-bottom: 8px; }
.la__ops { display: flex; flex-direction: column; gap: 18px; font-size: 22px; color: var(--azul-claro); text-align: center; }
.la__ops code { font-size: 18px; }
</style>
