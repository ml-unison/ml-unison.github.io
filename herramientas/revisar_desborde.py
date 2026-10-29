"""Busca láminas cuyo contenido se sale del área visible o tapa el pie.

Uso: uv run python herramientas/revisar_desborde.py [filtro] [--capturas DIR]
Construye cada presentación (Slidev) en una carpeta temporal, la abre con Chrome
(playwright) con todos los clics revelados y mide los elementos de cada lámina.
"""
import subprocess, sys, tempfile, threading, functools, http.server, socketserver
from pathlib import Path
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parent.parent
P = RAIZ / 'presentaciones'
args = [a for a in sys.argv[1:] if not a.startswith('--')]
filtro = args[0] if args else ''
capturas = Path(sys.argv[sys.argv.index('--capturas') + 1]) if '--capturas' in sys.argv else None
if capturas and str(capturas) in args: args.remove(str(capturas))

MEDIR = """() => {
  const lay = document.querySelector('.slidev-page-' + %d + ' .slidev-layout') || document.querySelector('.slidev-layout');
  if (!lay) return null;
  const L = lay.getBoundingClientRect();
  const pie = document.querySelector('.slidev-page-' + %d + ' .pie');
  if (!pie) return null;   // portadas y separadores: solo decoración, sin pie
  const limite = pie.getBoundingClientRect().top;
  let peor = 0, quien = '';
  for (const e of lay.querySelectorAll('*')) {
    const r = e.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (e.closest('svg') && e.tagName.toLowerCase() !== 'svg') continue;  // lo de adentro lo recorta el propio SVG
    if (e.closest('.katex-mathml')) continue;                             // MathML oculto de KaTeX
    const st = getComputedStyle(e);
    if (st.visibility === 'hidden' || st.opacity === '0' || st.position === 'fixed') continue;
    const sale = Math.max(r.bottom - limite, r.right - L.right);
    if (sale > peor) { peor = sale; quien = (e.tagName + ' ' + (e.textContent || '').trim().slice(0, 60)); }
  }
  return [Math.round(peor), quien];
}"""

def servir(directorio):
    class Callado(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass
    h = functools.partial(Callado, directory=str(directorio))
    s = socketserver.TCPServer(('127.0.0.1', 0), h)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s

decks = sorted(f for f in P.glob('u[0-9][0-9]-*.md') if filtro in f.name)
problemas = 0
with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
    nav = pw.chromium.launch(channel='chrome')
    pag = nav.new_page(viewport={'width': 1280, 'height': 720})
    for f in decks:
        out = Path(tmp) / f.stem
        r = subprocess.run(['npx', 'slidev', 'build', f.name, '--base', '/', '--out', str(out)], cwd=P, capture_output=True, text=True)
        if r.returncode:
            print(f'{f.name}: NO COMPILA\n{r.stdout[-800:]}{r.stderr[-800:]}'); problemas += 1; continue
        srv = servir(out); puerto = srv.server_address[1]
        pag.goto(f'http://127.0.0.1:{puerto}/#/2')
        pag.wait_for_timeout(1500)
        n = int(pag.evaluate("() => (document.querySelector('.pie')?.textContent.match(/(\\d+)\\s*$/) || [0, 0])[1]"))
        i, vistos = 1, 0
        while i <= n:
            pag.goto(f'http://127.0.0.1:{puerto}/#/{i}?clicks=99')
            pag.wait_for_timeout(700)
            res = pag.evaluate(MEDIR % (i, i))
            if res and res[0] > 10:
                problemas += 1
                print(f'{f.stem} lámina {i}: se sale {res[0]} px — {res[1]}')
                if capturas:
                    capturas.mkdir(parents=True, exist_ok=True)
                    pag.screenshot(path=str(capturas / f'{f.stem}-{i:02d}.png'))
            i += 1; vistos += 1
        srv.shutdown(); srv.server_close()
        print(f'{f.stem}: {vistos} láminas revisadas')
    nav.close()
print(f'{problemas} problemas' if problemas else 'Sin desbordes')
