// Construye el sitio completo en _construccion/:
//   - el sitio (Astro)                       → _construccion/
//   - cada presentación (Slidev)              → _construccion/presentaciones/<id>/
//   - exámenes archivados y quizzes exportados → se copian tal cual
//
// Variables de entorno:
//   BASE      "/" (producción, por omisión) o "/preview/" (vista previa)
//   RDP_RAMA  rama de la que Colab abre las libretas ("main" por omisión)
//
//   node herramientas/construir.mjs                 # todo
//   node herramientas/construir.mjs u01-01          # solo las presentaciones que coincidan (y el sitio)
import { execFileSync } from 'node:child_process'
import { cpSync, existsSync, readdirSync, rmSync } from 'node:fs'
import { join, resolve } from 'node:path'

const RAIZ = resolve(import.meta.dirname, '..')
const SALIDA = join(RAIZ, '_construccion')
const BASE = (process.env.BASE ?? '/').replace(/\/?$/, '/')
const filtro = process.argv.slice(2)

const correr = (cmd, args, cwd, env = {}) =>
  execFileSync(cmd, args, { cwd, stdio: 'inherit', env: { ...process.env, ...env } })

rmSync(SALIDA, { recursive: true, force: true })

console.log(`\n▸ Sitio (base ${BASE})`)
correr('npx', ['astro', 'build'], join(RAIZ, 'sitio'), { BASE, SALIDA })

const decks = readdirSync(join(RAIZ, 'presentaciones'))
  .filter(f => /^u\d\d-.*\.md$/.test(f))
  .filter(f => !filtro.length || filtro.some(t => f.includes(t)))
for (const f of decks) {
  const id = f.replace(/\.md$/, '')
  console.log(`\n▸ Presentación ${id}`)
  correr('npx', ['slidev', 'build', f, '--base', `${BASE}presentaciones/${id}/`, '--out', join(SALIDA, 'presentaciones', id)],
    join(RAIZ, 'presentaciones'))
}

for (const [origen, destino] of [['examenes/archivo', 'examenes/archivo'], ['quizzes/exportados', 'quizzes/exportados']]) {
  if (existsSync(join(RAIZ, origen)))
    cpSync(join(RAIZ, origen), join(SALIDA, destino), { recursive: true, filter: s => !/\.(tex|md|png)$/.test(s) || s.endsWith('.pdf') })
}
console.log(`\n✓ Listo: ${SALIDA}`)
