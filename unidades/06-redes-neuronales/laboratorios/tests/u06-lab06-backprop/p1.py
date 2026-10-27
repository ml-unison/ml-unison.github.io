OK_FORMAT = True

test = {   'name': 'p1',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.allclose(lineal_adelante(_X, _W, _b), [[5.5, 2.0, -1.5], [11.5, 4.0, -3.5], [17.5, 6.0, -5.5]]), 'Z = X W + b; el renglón 1 debe ser "
                                               "(5.5, 2, −1.5)'\n"
                                               '>>> _dX, _dW, _db = lineal_atras(np.ones((3, 3)), _X, _W)\n'
                                               ">>> assert np.shape(_dX) == (3, 2) and np.shape(_dW) == (2, 3) and (np.shape(_db) == (3,)), 'dX tiene la forma de X (N × d), dW la de W (d × m) y db "
                                               "la de b (m)'\n"
                                               ">>> assert np.allclose(_db, [3.0, 3.0, 3.0]), 'db suma dZ sobre los ejemplos (los renglones)'\n"
                                               ">>> assert np.allclose(_dW, [[9.0, 9.0, 9.0], [12.0, 12.0, 12.0]]), 'dW = Xᵀ dZ'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
