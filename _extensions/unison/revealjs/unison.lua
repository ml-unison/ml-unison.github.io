-- Filtro de las presentaciones UNISON:
--  * fondo azul marino en los separadores de sección (encabezados de nivel 1)
--  * escudo en la portada (el logo de cada diapositiva viene de _brand.yml)
-- Las rutas se calculan respecto a la raíz del proyecto, sin importar
-- en qué carpeta esté la presentación.

local MARINO = "#20305A"

local function ruta_marca(archivo)
  local offset = quarto.project.offset or "."
  return offset .. "/assets/brand/" .. archivo
end

function Header(h)
  if h.level == 1
     and not h.attributes["background-color"]
     and not h.attributes["background-image"] then
    h.attributes["background-color"] = MARINO
    h.classes:insert("rdp-seccion")
  end
  return h
end

function Meta(m)
  if m["rdp-sin-escudo"] then return m end
  local attrs = m["title-slide-attributes"] or {}
  if not attrs["data-background-image"] then
    attrs["data-background-image"] = ruta_marca("escudo-unison-color.png")
    attrs["data-background-size"] = "auto 46%"
    attrs["data-background-position"] = "92% 50%"
    m["title-slide-attributes"] = attrs
  end
  return m
end

return { { Meta = Meta }, { Header = Header } }
