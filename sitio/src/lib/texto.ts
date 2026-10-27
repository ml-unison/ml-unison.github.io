import katex from 'katex'

const escapar = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/**
 * Markdown mínimo para textos de los bancos de preguntas: $…$, $$…$$, **negritas**, *itálicas* y `código`.
 * Las fórmulas se convierten a HTML con KaTeX durante la construcción (sin JavaScript en el navegador).
 */
export function texto(md: string): string {
  const formulas: string[] = []
  let s = md.trim()
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, f) => { formulas.push(katex.renderToString(f, { displayMode: true, throwOnError: false })); return `\u0000${formulas.length - 1}\u0000` })
    .replace(/\$([^$\n]+?)\$/g, (_, f) => { formulas.push(katex.renderToString(f, { throwOnError: false })); return `\u0000${formulas.length - 1}\u0000` })
  s = escapar(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, '$1<em>$2</em>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, ' ')
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => formulas[Number(i)])
}
