OK_FORMAT = True

test = {   'name': 'p2',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [10.0, 10.0]])\n'
                                               '>>> _N = actualizar(_X, np.array([0, 0, 0, 1]), np.zeros((2, 2)))\n'
                                               ">>> assert np.allclose(_N, [[4.0, 0.0], [10.0, 10.0]]), 'cada centroide debe ser la media de su grupo (no la mediana)'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])\n'
                                               '>>> _C = np.array([[0.0, 0.0], [7.0, 7.0], [-3.0, 4.0]])\n'
                                               '>>> _N = actualizar(_X, np.array([0, 0, 1]), _C)\n'
                                               ">>> assert _N.shape == (3, 2), 'debe haber una fila por cada centroide de C'\n"
                                               ">>> assert np.allclose(_N[2], [-3.0, 4.0]), 'un grupo vacío debe conservar su centroide anterior'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
