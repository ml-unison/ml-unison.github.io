-- Shortcode {{< colab ruta/a/libreta.ipynb [texto] >}}
-- Genera una insignia "Abrir en Colab" que apunta a la rama configurada en curso.rama
local function meta_str(meta, clave, defecto)
  local curso = meta["curso"]
  if curso and curso[clave] then
    return pandoc.utils.stringify(curso[clave])
  end
  return defecto
end

return {
  ["colab"] = function(args, kwargs, meta, raw_args, context)
    local ruta = pandoc.utils.stringify(args[1])
    local texto = args[2] and pandoc.utils.stringify(args[2]) or "Abrir en Colab"
    local repo = meta_str(meta, "repo", "ml-unison/ml-unison.github.io")
    local rama = meta_str(meta, "rama", "main")
    local url = "https://colab.research.google.com/github/" .. repo .. "/blob/" .. rama .. "/" .. ruta
    local html = '<a class="rdp-colab" href="' .. url .. '" target="_blank" rel="noopener">' ..
      '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="' .. texto .. '"></a>'
    if context == "text" then
      -- dentro de bloques {=html} se espera texto plano
      return html
    end
    if quarto.doc.is_format("html") then
      return pandoc.RawInline("html", html)
    end
    return pandoc.Link(texto, url)
  end
}
