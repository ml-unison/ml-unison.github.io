OK_FORMAT = True

test = {   'name': 'p1',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _w0, _w = ridge(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]), 0.0)\n'
                                               ">>> assert abs(_w0 - 1.0) < 1e-09 and np.allclose(_w, [2.0]), 'con lam = 0 y los puntos (0, 1), (1, 3), (2, 5) debe dar w0 = 1, w = [2]'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _sk = Ridge(alpha=10.0).fit(X_ent, y_ent)\n'
                                               '>>> _w0, _w = ridge(X_ent, y_ent, 10.0)\n'
                                               ">>> assert np.shape(_w) == (20,), 'w debe tener un peso por columna (sin el sesgo)'\n"
                                               ">>> assert np.allclose(_w, _sk.coef_) and abs(_w0 - _sk.intercept_) < 1e-08, 'no coincide con Ridge de scikit-learn'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
