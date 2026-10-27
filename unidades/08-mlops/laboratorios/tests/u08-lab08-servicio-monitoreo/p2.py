OK_FORMAT = True

test = {   'name': 'p2',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert abs(psi(np.arange(100), np.arange(100), k=4)) < 1e-12, 'dos muestras iguales tienen PSI = 0'\n"
                                               '>>> _q = np.full(4, 0.25)\n'
                                               '>>> _p = np.array([0.5, 0.0, 0.25, 0.25])\n'
                                               '>>> _esperado = np.sum((np.maximum(_p, 0.0001) - _q) * np.log(np.maximum(_p, 0.0001) / _q))\n'
                                               '>>> _obtenido = psi(np.arange(100), np.array([-5.0, 10.0, 60.0, 500.0]), k=4)\n'
                                               ">>> assert np.isfinite(_obtenido), 'con una cubeta vacía el PSI debe ser finito: usa el épsilon'\n"
                                               ">>> assert abs(_obtenido - _esperado) < 1e-09, f'el PSI de este ejemplo es {_esperado:.4f}'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
