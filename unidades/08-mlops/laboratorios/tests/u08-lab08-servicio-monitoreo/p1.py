OK_FORMAT = True

test = {   'name': 'p1',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _q, _p = proporciones(np.arange(100), np.arange(100), k=4)\n'
                                               ">>> assert np.allclose(_q, [0.25, 0.25, 0.25, 0.25]) and np.allclose(_p, _q), 'con 0, 1, ..., 99 y k = 4 hay cuatro cubetas de 25 % en ambas "
                                               "muestras'\n"
                                               '>>> _q, _p = proporciones(np.arange(100), np.array([-5.0, 10.0, 60.0, 500.0]), k=4)\n'
                                               ">>> assert np.allclose(_p, [0.5, 0.0, 0.25, 0.25]), 'los cortes salen de la referencia (24.75, 49.5, 74.25); -5 y 10 caen en la primera cubeta y 500 "
                                               "en la última'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> _q, _p = proporciones(b2011['clima'], b2012['clima'])\n"
                                               ">>> assert len(_q) == len(_p) and abs(_q.sum() - 1) < 1e-12 and (abs(_p.sum() - 1) < 1e-12), 'las fracciones deben sumar 1'\n"
                                               ">>> assert len(_q) < 10, 'con un atributo discreto los cortes repetidos se eliminan y hay menos de k cubetas'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
