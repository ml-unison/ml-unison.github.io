-- Shortcode {{< quiz u01 >}}  ·  {{< quiz u01 ids="u01-p01,u01-p03" >}}
--
-- Lee quizzes/banco/<id>.yml (desde la raíz del proyecto) y genera un
-- cuestionario de autoevaluación en HTML. Los textos se interpretan como
-- Markdown, así que admiten $\LaTeX$, código y énfasis.
--
-- Esquema de cada pregunta:
--   id, tipo (opcion | multiple | vf | numerica), enunciado, explicacion,
--   opciones: [{texto, correcta, retro}]       (opcion, multiple)
--   respuesta: true|false                       (vf)
--   respuesta: número, tolerancia: número       (numerica)

local dependencia_agregada = false

local function agregar_dependencias()
  if dependencia_agregada then return end
  quarto.doc.add_html_dependency({
    name = "rdp-quiz",
    version = "0.1.0",
    scripts = { "quiz.js" },
    stylesheets = { "quiz.css" },
  })
  dependencia_agregada = true
end

local function escapar(s)
  return (s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"):gsub('"', "&quot;"))
end

local function a_html(valor)
  if valor == nil then return "" end
  local bloques
  local t = pandoc.utils.type(valor)
  if t == "Blocks" then
    bloques = valor
  elseif t == "Inlines" then
    bloques = pandoc.Blocks({ pandoc.Plain(valor) })
  else
    bloques = pandoc.read(pandoc.utils.stringify(valor), "markdown").blocks
  end
  return pandoc.write(pandoc.Pandoc(bloques), "html", { html_math_method = "mathjax" })
end

local function texto(valor)
  if valor == nil then return "" end
  return pandoc.utils.stringify(valor)
end

local function verdadero(valor)
  if valor == nil then return false end
  if type(valor) == "boolean" then return valor end
  local s = texto(valor):lower()
  return s == "true" or s == "verdadero" or s == "sí" or s == "si"
end

local function leer_banco(nombre)
  local raiz = quarto.project.directory or "."
  local ruta = pandoc.path.join({ raiz, "quizzes", "banco", nombre .. ".yml" })
  local f = io.open(ruta, "r")
  if not f then
    error("quiz: no encuentro el banco " .. ruta)
  end
  local contenido = f:read("a")
  f:close()
  -- Truco: un documento Markdown con solo bloque YAML = metadatos ya interpretados
  local doc = pandoc.read("---\n" .. contenido .. "\n---\n", "markdown")
  return doc.meta
end

local function render_opciones(p, idp, multiple)
  local partes = {}
  local tipo_input = multiple and "checkbox" or "radio"
  for i, op in ipairs(p.opciones or {}) do
    local idop = idp .. "-o" .. i
    table.insert(partes, string.format(
      '<label class="rdp-quiz-opcion" for="%s">' ..
      '<input type="%s" id="%s" name="%s" value="%d" data-correcta="%s"> ' ..
      '<span class="rdp-quiz-texto">%s</span>' ..
      '<span class="rdp-quiz-retro" hidden>%s</span></label>',
      idop, tipo_input, idop, idp, i, verdadero(op.correcta) and "1" or "0",
      a_html(op.texto), a_html(op.retro)))
  end
  return table.concat(partes, "\n")
end

local function render_pregunta(p, n, prefijo)
  local tipo = texto(p.tipo)
  if tipo == "" then tipo = "opcion" end
  local idp = prefijo .. "-" .. (texto(p.id) ~= "" and texto(p.id) or tostring(n))
  local cuerpo
  local datos = string.format('data-tipo="%s"', tipo)

  if tipo == "opcion" or tipo == "multiple" then
    cuerpo = render_opciones(p, idp, tipo == "multiple")
  elseif tipo == "vf" then
    local correcta = verdadero(p.respuesta)
    cuerpo = render_opciones({ opciones = {
      { texto = "Verdadero", correcta = correcta },
      { texto = "Falso", correcta = not correcta },
    } }, idp, false)
  elseif tipo == "numerica" then
    datos = datos .. string.format(' data-respuesta="%s" data-tolerancia="%s"',
      escapar(texto(p.respuesta)), escapar(texto(p.tolerancia) ~= "" and texto(p.tolerancia) or "0"))
    cuerpo = string.format(
      '<input class="rdp-quiz-numero" type="text" inputmode="decimal" id="%s-n" aria-label="Respuesta numérica">', idp)
  else
    error("quiz: tipo de pregunta desconocido '" .. tipo .. "' en " .. idp)
  end

  local etiqueta = (tipo == "multiple") and '<span class="rdp-quiz-tipo">Selecciona todas las correctas</span>' or ""
  return string.format(
    '<fieldset class="rdp-quiz-pregunta" id="%s" %s>' ..
    '<legend><span class="rdp-quiz-num">%d</span> %s</legend>%s' ..
    '<div class="rdp-quiz-cuerpo">%s</div>' ..
    '<div class="rdp-quiz-acciones"><button type="button" class="rdp-quiz-verificar">Verificar</button>' ..
    '<span class="rdp-quiz-resultado" role="status" aria-live="polite"></span></div>' ..
    '<div class="rdp-quiz-explicacion" hidden>%s</div></fieldset>',
    idp, datos, n, a_html(p.enunciado), etiqueta, cuerpo, a_html(p.explicacion))
end

return {
  ["quiz"] = function(args, kwargs)
    if not quarto.doc.is_format("html") then
      return pandoc.Para({ pandoc.Emph("(Cuestionario interactivo disponible en la versión web.)") })
    end
    agregar_dependencias()
    local nombre = texto(args[1])
    local banco = leer_banco(nombre)

    local filtro = nil
    local ids = texto(kwargs["ids"])
    if ids ~= "" then
      filtro = {}
      for id in ids:gmatch("[^,%s]+") do filtro[id] = true end
    end

    local partes = {}
    local n = 0
    for _, p in ipairs(banco.preguntas or {}) do
      if not filtro or filtro[texto(p.id)] then
        n = n + 1
        table.insert(partes, render_pregunta(p, n, "quiz-" .. nombre))
      end
    end

    local titulo = texto(banco.titulo)
    local html = string.format(
      '<section class="rdp-quiz" data-banco="%s">' ..
      '<header class="rdp-quiz-encabezado"><span class="rdp-quiz-insignia">Autoevaluación</span>' ..
      '<h3 class="rdp-quiz-titulo anchored">%s</h3>' ..
      '<p class="rdp-quiz-marcador" aria-live="polite">0 de %d correctas</p></header>%s' ..
      '<footer class="rdp-quiz-pie"><button type="button" class="rdp-quiz-reiniciar">Reiniciar</button></footer></section>',
      escapar(nombre), escapar(titulo), n, table.concat(partes, "\n"))
    return pandoc.RawBlock("html", html)
  end
}
