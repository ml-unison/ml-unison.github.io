OK_FORMAT = True

test = {   'name': 'p3',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': ">>> assert abs(ks_estadistico([1, 2, 3], [1, 2, 3])) < 1e-12, 'dos muestras iguales tienen D = 0'\n"
                                               ">>> assert abs(ks_estadistico([1, 2, 3], [10, 11])) == 1.0, 'si todos los datos de una muestra son menores, D = 1'\n"
                                               ">>> assert abs(ks_estadistico([1, 2, 3, 4], [3, 4, 5, 6]) - 0.5) < 1e-12, 'en x = 2, F_a = 1/2 y F_b = 0, y no hay mayor distancia'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _rng = np.random.default_rng(3)\n'
                                               '>>> _a, _b = (_rng.normal(size=300), _rng.normal(0.3, 1, size=200))\n'
                                               ">>> assert abs(ks_estadistico(_a, _b) - stats.ks_2samp(_a, _b).statistic) < 1e-12, 'no coincide con scipy.stats.ks_2samp'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
