OK_FORMAT = True

test = {   'name': 'p3',
    'points': 1,
    'suites': [   {   'cases': [   {   'code': ">>> assert abs(inercia(np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([0, 0])) - 2.0) < 1e-09, 'dos puntos a distancia 1 de su "
                                               "centroide: la inercia debe ser 2'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _C = X[:5]\n'
                                               '>>> _e = asignar(X, _C)\n'
                                               ">>> assert abs(inercia(X, _C, _e) - ((X - _C[_e]) ** 2).sum()) < 1e-06, 'revisa la suma de distancias al cuadrado'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
