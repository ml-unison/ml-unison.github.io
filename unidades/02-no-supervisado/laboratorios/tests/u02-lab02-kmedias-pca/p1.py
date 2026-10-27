OK_FORMAT = True

test = {   'name': 'p1',
    'points': 1,
    'suites': [   {   'cases': [   {   'code': '>>> _X = np.array([[0.0, 0.0], [5.0, 5.0], [4.0, 6.0], [-1.0, 0.0]])\n'
                                               '>>> _C = np.array([[0.0, 1.0], [5.0, 5.0]])\n'
                                               ">>> assert list(asignar(_X, _C)) == [0, 1, 1, 0], 'revisa las distancias: el resultado para este ejemplo debe ser [0, 1, 1, 0]'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {'code': ">>> assert asignar(X, X[:5]).shape == (600,), 'asignar debe devolver un arreglo con un índice por punto'\n", 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
