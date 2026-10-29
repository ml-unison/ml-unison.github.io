// Revisa que todas las fórmulas de las presentaciones compilen con KaTeX y las macros del curso.
// Uso: node herramientas/revisar_katex.mjs [filtro]   (p. ej. u03)
import { readFileSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
const RAIZ = join(dirname(fileURLToPath(import.meta.url)), '..')
const P = join(RAIZ, 'presentaciones')
const katex = createRequire(join(P, 'package.json'))('katex')
const src = readFileSync(join(P, 'setup/katex.ts'), 'utf8')
const macros = {}
for (const m of src.matchAll(/'(\\\\[A-Za-z]+)':\s*'([^']*)'/g)) macros[m[1].replace(/\\\\/g, '\\')] = m[2].replace(/\\\\/g, '\\')
const filtro = process.argv[2] ?? ''
let errores = 0
for (const f of readdirSync(P).filter(f => /^u\d\d-.*\.md$/.test(f) && f.includes(filtro))) {
  const t = readFileSync(join(P, f), 'utf8').replace(/```[\s\S]*?```/g, '')
  const formulas = [...t.matchAll(/\$\$([\s\S]+?)\$\$/g)].map(m => [m[1], true])
  const sin = t.replace(/\$\$[\s\S]+?\$\$/g, '')
  for (const m of sin.matchAll(/(?<![\\$])\$([^$\n]+?)\$/g)) formulas.push([m[1], false])
  for (const [tex, display] of formulas) {
    try { katex.renderToString(tex, { displayMode: display, throwOnError: true, strict: false, macros: { ...macros } }) }
    catch (e) { errores++; console.log(`${f}: ${e.message.split('\n')[0]}\n    ${tex.trim().slice(0, 120)}`) }
  }
}
console.log(errores ? `${errores} errores` : 'Sin errores de KaTeX')
process.exit(errores ? 1 : 0)
