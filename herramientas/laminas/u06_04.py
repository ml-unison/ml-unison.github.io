"""Datos de las láminas u06-04 (secuencias, recurrentes y atención).

Reproduce, con el mismo código y las mismas semillas, los cálculos de la libreta
unidades/06-redes-neuronales/ejemplos/u06-e04-secuencias-atencion.py: n-gramas,
LSTM y transformer de caracteres sobre el Quijote. Tarda unos 3 minutos en CPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _comun import r  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
SEMILLA = 2011
T = 128
CONTEXTO_MIN = 64
B = 32
CALENTAMIENTO = 50
RECORTE = 1.0


def datos() -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dispositivo = torch.device("cpu")
    torch.manual_seed(SEMILLA)

    texto = (RAIZ / "datos" / "quijote.txt").read_text(encoding="utf-8")
    vocabulario = sorted(set(texto))
    V = len(vocabulario)
    a_indice = {c: i for i, c in enumerate(vocabulario)}
    ids = np.array([a_indice[c] for c in texto], dtype=np.int64)
    corte = int(0.9 * len(ids))
    ids_tr, ids_va = ids[:corte], ids[corte:]
    inicios_va = np.arange(0, len(ids_va) - T, CONTEXTO_MIN)
    objetivos_va = (inicios_va[:, None] + np.arange(CONTEXTO_MIN + 1, T + 1)).ravel()
    out: dict = {"n_caracteres": len(texto), "V": V, "n_tr": len(ids_tr), "n_va": len(ids_va),
                 "n_calificadas": len(objetivos_va)}

    def perplejidad(lp):
        return float(np.exp(-np.mean(lp)))

    # ===== n-gramas (sección 3 de la libreta)
    def codigos(s, k):
        c = np.zeros(len(s) - k + 1, dtype=np.int64)
        for j in range(k):
            c = c * V + s[j:len(s) - k + 1 + j]
        return c

    class NGramas:
        def __init__(self, n, alfa=1.0):
            self.n, self.alfa = n, alfa
            self.gramas, self.n_gramas = np.unique(codigos(ids_tr, n), return_counts=True)
            if n > 1:
                self.contextos, self.n_contextos = np.unique(codigos(ids_tr[:-1], n - 1), return_counts=True)

        @staticmethod
        def _contar(tabla, cuentas, claves):
            i = np.minimum(np.searchsorted(tabla, claves), len(tabla) - 1)
            return np.where(tabla[i] == claves, cuentas[i], 0)

        def log_prob(self, s, posiciones):
            g = codigos(s, self.n)[posiciones - self.n + 1]
            ng = self._contar(self.gramas, self.n_gramas, g)
            nc = self._contar(self.contextos, self.n_contextos, g // V) if self.n > 1 else len(ids_tr)
            return np.log((ng + self.alfa) / (nc + self.alfa * V))

    p_uni = (np.bincount(ids_tr, minlength=V) + 1) / (len(ids_tr) + V)
    pos_tr = np.arange(6, len(ids_tr))
    bigrama = NGramas(2, 1.0)
    ng = {"n": list(range(1, 8))}
    for alfa, k in [(1.0, "a1"), (0.01, "a001")]:
        tr_, va_, nunca, distintos = [], [], [], []
        for n in range(1, 8):
            m = NGramas(n, alfa)
            tr_.append(perplejidad(m.log_prob(ids_tr, pos_tr)))
            va_.append(perplejidad(m.log_prob(ids_va, objetivos_va)))
            nunca.append(100 * float(np.mean(NGramas._contar(m.gramas, m.n_gramas, codigos(ids_va, n)[objetivos_va - n + 1]) == 0)))
            distintos.append(len(m.gramas))
        ng[k] = {"tr": r(tr_, 2), "va": r(va_, 2)}
        ng["nunca"], ng["distintos"] = r(nunca, 1), distintos
    out["ngramas"] = ng
    mejor = NGramas(5, 0.01)
    pp_uni = perplejidad(np.log(p_uni[ids_va[objetivos_va]]))
    pp_bigrama = perplejidad(bigrama.log_prob(ids_va, objetivos_va))
    pp_ngrama = perplejidad(mejor.log_prob(ids_va, objetivos_va))

    # distribución del bigrama tras «q» y tras el espacio
    def siguiente(letra):
        pares = np.column_stack([np.full(V, a_indice[letra]), np.arange(V)]).ravel()
        p = np.exp(bigrama.log_prob(pares, np.arange(1, 2 * V, 2)))
        orden = np.argsort(p)[::-1][:12]
        return {"letras": [vocabulario[i] for i in orden], "p": r(p[orden], 4),
                "ppl": round(float(np.exp(-np.sum(p * np.log(p)))), 2)}
    out["siguiente"] = {"q": siguiente("q"), "espacio": siguiente(" ")}

    # ===== RNN: producto de jacobianas (sección 4)
    def norma_jacobiana_rnn(rho, m=256, pasos=100, semilla=SEMILLA):
        g = torch.Generator().manual_seed(semilla)
        Q, _ = torch.linalg.qr(torch.randn(m, m, generator=g))
        W = rho * Q
        entradas = 0.5 * torch.randn(pasos, m, generator=g)
        h, D = torch.zeros(m), []
        for t in range(pasos):
            h = torch.tanh(W @ h + entradas[t])
            D.append(1 - h ** 2)
        J, normas = torch.eye(m), [1.0]
        for s in range(pasos - 1, 0, -1):
            J = J @ (D[s][:, None] * W)
            normas.append((J.norm() / math.sqrt(m)).item())
        return np.array(normas)
    out["rnn"] = {str(rho): r(np.log10(norma_jacobiana_rnn(rho)), 3) for rho in (0.8, 1.0, 1.5, 3.0)}

    # ===== entrenamiento común (sección 5)
    datos_va = torch.from_numpy(ids_va).to(dispositivo)

    def lote(generador):
        i = torch.randint(0, len(ids_tr) - T - 1, (B,), generator=generador)
        v = torch.stack([torch.from_numpy(ids_tr[j:j + T + 1]) for j in i.tolist()]).to(dispositivo)
        return v[:, :-1], v[:, 1:]

    @torch.no_grad()
    def evaluar(modelo, ventanas=None):
        modelo.eval()
        inicios = inicios_va if ventanas is None else inicios_va[ventanas]
        total = 0.0
        for k in range(0, len(inicios), 128):
            v = torch.stack([datos_va[j:j + T + 1] for j in inicios[k:k + 128]])
            logits = modelo(v[:, :-1])[:, -CONTEXTO_MIN:]
            total += F.cross_entropy(logits.reshape(-1, V), v[:, -CONTEXTO_MIN:].reshape(-1), reduction="sum").item()
        modelo.train()
        return math.exp(total / (len(inicios) * CONTEXTO_MIN))

    def entrenar(modelo, pasos, tasa, cada=100):
        opt = torch.optim.AdamW(modelo.parameters(), lr=tasa, weight_decay=0.01)
        calendario = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: min(1.0, (t + 1) / CALENTAMIENTO) * 0.5 * (1 + math.cos(math.pi * t / pasos)))
        generador = torch.Generator().manual_seed(SEMILLA)
        muestra_va = np.arange(0, len(inicios_va), 8)
        historia, normas = [], []
        for paso in range(1, pasos + 1):
            x, y = lote(generador)
            perdida = F.cross_entropy(modelo(x).reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            perdida.backward()
            normas.append(nn.utils.clip_grad_norm_(modelo.parameters(), RECORTE).item())
            opt.step()
            calendario.step()
            if paso % cada == 0 or paso == pasos:
                historia.append([paso, math.exp(perdida.item()), evaluar(modelo, muestra_va)])
        return np.array(historia), np.array(normas)

    class LSTMLenguaje(nn.Module):
        def __init__(self, V, d=64, m=256):
            super().__init__()
            self.encaje = nn.Embedding(V, d)
            self.lstm = nn.LSTM(d, m, batch_first=True)
            self.salida = nn.Linear(m, V)

        def forward(self, x):
            h, _ = self.lstm(self.encaje(x))
            return self.salida(h)

    torch.manual_seed(SEMILLA)
    lstm = LSTMLenguaje(V).to(dispositivo)
    lstm_inicial = LSTMLenguaje(V).to(dispositivo)
    lstm_inicial.load_state_dict(lstm.state_dict())
    historia_lstm, normas_lstm = entrenar(lstm, 1200, tasa=0.02)
    pp_lstm = evaluar(lstm)

    def gradiente_por_distancia(modelo, n_ventanas=64):
        modelo.eval()
        v = torch.stack([datos_va[j:j + T + 1] for j in inicios_va[:n_ventanas]])
        e = modelo.encaje(v[:, :-1]).detach().requires_grad_(True)
        if isinstance(modelo, LSTMLenguaje):
            logits = modelo.salida(modelo.lstm(e)[0])
        else:
            logits = modelo.desde_encajes(e)
        F.cross_entropy(logits[:, -1], v[:, -1], reduction="sum").backward()
        modelo.train()
        return e.grad.norm(dim=-1).mean(0).flip(0).cpu().numpy() / n_ventanas

    def log_relativa(g):
        rel = g / g[0]
        return [None if x <= 0 else round(float(np.log10(x)), 3) for x in rel]

    g_ini, g_ent = gradiente_por_distancia(lstm_inicial), gradiente_por_distancia(lstm)

    # ===== escala √d_k (sección 6)
    gen = torch.Generator().manual_seed(SEMILLA)
    escala = {"dk": [4, 16, 64, 256, 1024], "max_sin": [], "max_con": [], "jac_sin": [], "jac_con": [], "de_sin": []}
    for dk in escala["dk"]:
        q = torch.randn(2000, 1, dk, generator=gen)
        k = torch.randn(2000, 64, dk, generator=gen)
        puntajes = (q * k).sum(-1)
        for esc, suf in [(1.0, "sin"), (math.sqrt(dk), "con")]:
            p = torch.softmax(puntajes / esc, -1)
            jac = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
            escala[f"max_{suf}"].append(round(p.max(-1).values.mean().item(), 3))
            escala[f"jac_{suf}"].append(round(jac.norm(dim=(1, 2)).mean().item(), 3))
            if suf == "sin":
                escala["de_sin"].append(round(puntajes.std().item(), 2))
    out["escala"] = escala

    # ===== transformer (sección 7)
    class AutoatencionCausal(nn.Module):
        def __init__(self, d, cabezas):
            super().__init__()
            self.cabezas, self.dk = cabezas, d // cabezas
            self.qkv = nn.Linear(d, 3 * d)
            self.proyeccion = nn.Linear(d, d)
            self.guardar_pesos, self.pesos = False, None

        def forward(self, x):
            B_, T_, d = x.shape
            q, k, v = self.qkv(x).split(d, dim=-1)
            q, k, v = (z.view(B_, T_, self.cabezas, self.dk).transpose(1, 2) for z in (q, k, v))
            puntajes = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
            futuro = torch.triu(torch.ones(T_, T_, dtype=torch.bool, device=x.device), diagonal=1)
            alfa = puntajes.masked_fill(futuro, float("-inf")).softmax(dim=-1)
            if self.guardar_pesos:
                self.pesos = alfa.detach()
            y = (alfa @ v).transpose(1, 2).reshape(B_, T_, d)
            return self.proyeccion(y)

    class BloqueTransformer(nn.Module):
        def __init__(self, d, cabezas):
            super().__init__()
            self.norma1, self.norma2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.atencion = AutoatencionCausal(d, cabezas)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x):
            x = x + self.atencion(self.norma1(x))
            return x + self.mlp(self.norma2(x))

    class TransformerLenguaje(nn.Module):
        def __init__(self, V, d=128, cabezas=4, bloques=2, contexto=T):
            super().__init__()
            self.encaje = nn.Embedding(V, d)
            self.posicion = nn.Embedding(contexto, d)
            self.bloques = nn.ModuleList([BloqueTransformer(d, cabezas) for _ in range(bloques)])
            self.norma = nn.LayerNorm(d)
            self.salida = nn.Linear(d, V)

        def desde_encajes(self, e):
            x = e + self.posicion(torch.arange(e.shape[1], device=e.device))
            for bloque in self.bloques:
                x = bloque(x)
            return self.salida(self.norma(x))

        def forward(self, x):
            return self.desde_encajes(self.encaje(x))

    torch.manual_seed(SEMILLA)
    transformer = TransformerLenguaje(V).to(dispositivo)
    historia_tf, _ = entrenar(transformer, 1500, tasa=0.008)
    pp_tf = evaluar(transformer)
    g_tf = gradiente_por_distancia(transformer)

    n_param = lambda m: sum(p.numel() for p in m.parameters())
    out["comparacion"] = [
        {"modelo": "uniforme", "param": 0, "ppl": float(V)},
        {"modelo": "unigrama", "param": V, "ppl": round(pp_uni, 2)},
        {"modelo": "bigrama (Laplace)", "param": V * V, "ppl": round(pp_bigrama, 2)},
        {"modelo": "5-grama (α = 0.01)", "param": len(mejor.gramas), "ppl": round(pp_ngrama, 2)},
        {"modelo": "LSTM", "param": n_param(lstm), "ppl": round(pp_lstm, 2)},
        {"modelo": "transformer", "param": n_param(transformer), "ppl": round(pp_tf, 2)},
    ]
    out["curvas"] = {"lstm": {"paso": historia_lstm[:, 0].astype(int).tolist(), "va": r(historia_lstm[:, 2], 3)},
                     "tf": {"paso": historia_tf[:, 0].astype(int).tolist(), "va": r(historia_tf[:, 2], 3)},
                     "bigrama": round(pp_bigrama, 2), "ngrama": round(pp_ngrama, 2)}
    out["recorte"] = {"normas": r(normas_lstm, 3), "recortados": int(np.sum(normas_lstm > RECORTE)),
                      "mediana": round(float(np.median(normas_lstm)), 2), "maxima": round(float(normas_lstm.max()), 2),
                      "paso_max": int(normas_lstm.argmax() + 1)}
    out["gradiente"] = {"lstm_ini": log_relativa(g_ini), "lstm": log_relativa(g_ent), "tf": log_relativa(g_tf)}

    # ===== generación con temperatura (sección 9)
    @torch.no_grad()
    def generar(modelo, inicio, n=300, temperatura=1.0, semilla=SEMILLA):
        modelo.eval()
        g = torch.Generator(device=dispositivo).manual_seed(semilla)
        x = torch.tensor([a_indice[c] for c in inicio], device=dispositivo)
        for _ in range(n):
            logits = modelo(x[-T:][None])[0, -1]
            p = torch.softmax(logits / temperatura, dim=-1)
            x = torch.cat([x, torch.multinomial(p, 1, generator=g)])
        modelo.train()
        return "".join(vocabulario[i] for i in x.tolist())

    INICIO = "—Sancho amigo —dijo don Quijote—,"
    out["generados"] = {"inicio": INICIO, "textos": {str(t): generar(transformer, INICIO, temperatura=t) for t in (0.5, 0.8, 1.2)}}

    # ===== sorpresa por letra (sección 9)
    FRASE = "Hay mucho que decir en eso —dijo Sancho—. Durmamos, por ahora, entrambos"
    i0 = texto.index(FRASE, corte)
    ventana = torch.from_numpy(ids[i0 + len(FRASE) - T - 1:i0 + len(FRASE)]).to(dispositivo)   # la frase son las últimas letras
    with torch.no_grad():
        transformer.eval()
        lp = torch.log_softmax(transformer(ventana[None, :-1])[0], -1)
        transformer.train()
    sorpresa = -(lp[torch.arange(T), ventana[1:]] / math.log(2)).cpu().numpy()[-len(FRASE):]
    out["sorpresa"] = {"frase": FRASE, "bits": r(sorpresa, 2)}

    # ===== pesos de atención (sección 10)
    @torch.no_grad()
    def pesos_atencion(modelo, x):
        for bloque in modelo.bloques:
            bloque.atencion.guardar_pesos = True
        modelo.eval()
        modelo(x)
        modelo.train()
        for bloque in modelo.bloques:
            bloque.atencion.guardar_pesos = False
        return torch.stack([b.atencion.pesos for b in modelo.bloques], dim=1).cpu().numpy()

    pesos = pesos_atencion(transformer, ventana[None, 1:])[0]
    L = 40
    ultimas = "".join(vocabulario[i] for i in ventana[1:].tolist()[-L:])
    out["atencion"] = {"letras": ultimas, "pesos": [[r(pesos[b, h, -L:, -L:], 3) for h in range(4)] for b in range(2)]}

    v = torch.stack([datos_va[j:j + T] for j in inicios_va[:64]])
    P = pesos_atencion(transformer, v)[..., -CONTEXTO_MIN:, :]
    filas_t = np.arange(T - CONTEXTO_MIN, T)
    distancia = np.clip(filas_t[:, None] - np.arange(T)[None, :], 0, None)
    x = v.cpu().numpy()
    separador = np.isin(x, [a_indice[" "], a_indice["\n"]])
    tras_separador = np.zeros_like(separador)
    tras_separador[:, 1:] = separador[:, :-1]
    inicio_palabra = np.maximum.accumulate(np.where(tras_separador, np.arange(T), 0), axis=1)[:, filas_t]
    es_salto = (x == a_indice["\n"])[:, None, :]
    perfil = []
    for b in range(2):
        for h in range(4):
            A = P[:, b, h]
            atras = lambda k: round(float(A[:, np.arange(CONTEXTO_MIN), filas_t - k].mean()), 2)
            perfil.append({"bloque": b + 1, "cabeza": h + 1, "t1": atras(1), "t2": atras(2), "t3": atras(3),
                           "inicio": round(float(np.take_along_axis(A, inicio_palabra[..., None], -1).mean()), 2),
                           "saltos": round(float((A * es_salto).sum(-1).mean()), 2),
                           "distancia": round(float((A * distancia).sum(-1).mean()), 1)})
    out["perfil"] = perfil
    out["frac_saltos"] = round(float(np.mean(ids_va == a_indice["\n"])), 3)
    return out


if __name__ == "__main__":
    import json

    d = datos()
    print(json.dumps({k: d[k] for k in ("comparacion", "perfil", "recorte")}, ensure_ascii=False)[:3000])
