import { defineConfig } from 'astro/config'
import mdx from '@astrojs/mdx'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

// BASE: "/" en producción y "/preview/" en la vista previa (lo fija herramientas/construir.mjs)
const base = process.env.BASE ?? '/'

export default defineConfig({
  site: process.env.SITIO ?? 'https://ml-unison.github.io',
  base,
  trailingSlash: 'always',
  outDir: process.env.SALIDA ?? './dist',
  integrations: [mdx()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
})
