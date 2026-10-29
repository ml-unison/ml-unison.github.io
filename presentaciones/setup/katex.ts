import { defineKatexSetup } from '@slidev/types'

// Macros comunes a todas las presentaciones del curso
export default defineKatexSetup(() => ({
  macros: {
    '\\R': '\\mathbb{R}',
    '\\E': '\\mathbb{E}',
    '\\Prob': '\\mathbb{P}',
    '\\Var': '\\operatorname{Var}',
    '\\argmin': '\\operatorname*{arg\\,min}',
    '\\argmax': '\\operatorname*{arg\\,max}',
    '\\vx': '\\mathbf{x}',
    '\\vw': '\\mathbf{w}',
    '\\vy': '\\mathbf{y}',
    '\\va': '\\mathbf{a}',
    '\\vz': '\\mathbf{z}',
    '\\vX': '\\mathbf{X}',
    '\\vW': '\\mathbf{W}',
    '\\vth': '\\boldsymbol{\\theta}',
    '\\Hyp': '\\mathcal{H}',
    '\\Dat': '\\mathcal{D}',
    '\\X': '\\mathcal{X}',
    '\\Y': '\\mathcal{Y}',
    '\\Ein': 'E_{\\text{in}}',
    '\\Eout': 'E_{\\text{out}}',
    '\\T': '^{\\top}',
    '\\ind': '\\mathbb{1}\\!\\left[#1\\right]',
    '\\norm': '\\left\\lVert #1 \\right\\rVert',
  },
}))
