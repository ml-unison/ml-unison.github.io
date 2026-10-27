Un módulo por presentación (desde la Unidad 5): `uNN_MM.py` con `def datos() -> dict`.
`herramientas/datos_presentaciones.py uNN_MM` escribe `presentaciones/datos/uNN-MM.json`,
que los componentes importan con `import d from '../datos/uNN-MM.json'`.
Para no duplicar código, se pueden importar utilidades de `_comun.py`.
