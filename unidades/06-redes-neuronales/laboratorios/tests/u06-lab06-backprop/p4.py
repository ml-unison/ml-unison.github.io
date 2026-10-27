OK_FORMAT = True

test = {   'name': 'p4',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.allclose(gradiente_numerico(lambda v: np.sum(v ** 3), np.array([1.0, 2.0, -1.0])), [3.0, 12.0, 3.0], atol=1e-08), 'la derivada de Σ v³ "
                                               "es 3v²'\n"
                                               ">>> assert abs(error_relativo(np.array([1.0, 0.0]), np.array([1.0, 0.0]))) < 1e-15, 'dos vectores iguales tienen error relativo 0'\n"
                                               ">>> assert abs(error_relativo(np.array([3.0, 0.0]), np.array([0.0, 4.0])) - 5 / 7) < 1e-12, '‖(3, −4)‖ / (3 + 4) = 5/7'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _M = np.array([[1.0, 2.0], [3.0, 4.0]])\n'
                                               '>>> _M0 = _M.copy()\n'
                                               '>>> _g = gradiente_numerico(lambda A: np.sum(A ** 2) + A[0, 1] * A[1, 0], _M)\n'
                                               ">>> assert _g.shape == (2, 2), 'el gradiente tiene la forma de x'\n"
                                               ">>> assert np.allclose(_g, [[2.0, 7.0], [8.0, 8.0]], atol=1e-08), 'debe funcionar con matrices'\n"
                                               ">>> assert np.array_equal(_M, _M0), 'x debe quedar exactamente como estaba'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
