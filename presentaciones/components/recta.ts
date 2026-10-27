// Segmento de la recta w·x + b = c recortado al rectángulo [x0, x1] × [y0, y1] (coordenadas de datos).
export function recta(w: number[], b: number, c: number, [x0, x1, y0, y1]: number[]): number[][] | null {
  const pts: number[][] = []
  if (Math.abs(w[1]) > 1e-12) for (const x of [x0, x1]) { const y = (c - b - w[0] * x) / w[1]; if (y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]) }
  if (Math.abs(w[0]) > 1e-12) for (const y of [y0, y1]) { const x = (c - b - w[1] * y) / w[0]; if (x >= x0 - 1e-9 && x <= x1 + 1e-9) pts.push([x, y]) }
  return pts.length >= 2 ? [pts[0], pts[pts.length - 1]] : null
}
