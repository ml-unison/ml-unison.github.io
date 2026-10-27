OK_FORMAT = True

test = {   'name': 'p4',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _rep = reporte_cambio(b2011, b2012, VARIABLES)\n'
                                               ">>> assert list(_rep.columns) == ['psi', 'ks', 'nivel'], 'las columnas son psi, ks y nivel, en ese orden'\n"
                                               ">>> assert sorted(_rep.index) == sorted(VARIABLES), 'debe haber un renglón por atributo, con su nombre como índice'\n"
                                               ">>> assert np.all(np.diff(_rep['psi'].to_numpy()) <= 0), 'ordena de mayor a menor PSI'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> _rep = reporte_cambio(b2011, b2012, ['temp', 'hora'])\n"
                                               ">>> assert abs(_rep.loc['temp', 'psi'] - psi(b2011['temp'], b2012['temp'])) < 1e-12\n"
                                               ">>> assert abs(_rep.loc['hora', 'ks'] - ks_estadistico(b2011['hora'], b2012['hora'])) < 1e-12\n"
                                               ">>> assert _rep.loc['hora', 'nivel'] == 'estable', 'la distribución de las horas casi no cambia: nivel «estable»'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
