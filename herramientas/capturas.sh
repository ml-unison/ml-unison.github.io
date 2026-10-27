#!/usr/bin/env bash
# Capturas de diapositivas con Chrome sin interfaz, para revisión visual.
# Uso: herramientas/capturas.sh <url-de-la-presentación> <dir-salida> id1 id2 ...
# (requiere un servidor local, p. ej. `python3 -m http.server -d _site 8765`)
set -euo pipefail
URL="$1"; OUT="$2"; shift 2
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
mkdir -p "$OUT"
for id in "$@"; do
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size=1600,900 \
    --virtual-time-budget=10000 --screenshot="$OUT/$id.png" "$URL#/$id" >/dev/null 2>&1
  echo "$OUT/$id.png"
done
