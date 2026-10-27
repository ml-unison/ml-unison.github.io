<template>
  <div class="dp">
    <div class="dp__fila">
      <div class="dp__titulo">Programación clásica</div>
      <div class="dp__entradas">
        <div class="bloque">Datos</div>
        <div class="bloque">Reglas<small>escritas a mano</small></div>
      </div>
      <Flecha :largo="54" />
      <div class="bloque bloque--clave dp__centro">Programa</div>
      <Flecha :largo="54" />
      <div class="bloque bloque--fantasma dp__salida">Respuestas</div>
    </div>
    <div class="dp__fila">
      <div class="dp__titulo">Aprendizaje automático</div>
      <div class="dp__entradas">
        <div class="bloque">Datos</div>
        <div class="bloque">Respuestas<small>ejemplos etiquetados</small></div>
      </div>
      <Flecha :largo="54" />
      <div class="bloque bloque--clave dp__centro">Algoritmo de<br>aprendizaje</div>
      <Flecha :largo="54" />
      <div class="bloque bloque--resultado dp__salida">Modelo<small>las reglas, aprendidas</small></div>
    </div>
  </div>
</template>

<style scoped>
.dp { display: flex; flex-direction: column; gap: 30px; }
.dp__fila { display: grid; grid-template-columns: 210px 230px auto 230px auto 250px; align-items: center; column-gap: 14px; }
.dp__titulo { font-family: var(--f-titulo); font-weight: 650; font-size: 16px; letter-spacing: .08em; text-transform: uppercase; color: var(--azul); line-height: 1.3; }
.dp__entradas { display: flex; flex-direction: column; gap: 10px; border-right: 3px solid var(--linea); padding-right: 16px; }
.dp__entradas .bloque { padding: 10px 14px; }
.dp__centro, .dp__salida { padding: 22px 16px; }
</style>
