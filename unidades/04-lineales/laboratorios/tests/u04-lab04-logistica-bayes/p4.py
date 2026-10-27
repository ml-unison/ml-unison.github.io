OK_FORMAT = True

test = {   'name': 'p4',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': '>>> _w0, _w = newton_logistica(X_ent, y_ent)\n'
                                               ">>> assert np.shape(_w) == (5,), 'w debe tener un peso por variable (sin la intersección)'\n"
                                               '>>> _sk = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(X_ent, y_ent)\n'
                                               ">>> assert abs(_w0 - _sk.intercept_[0]) < 0.0001 and np.allclose(_w, _sk.coef_[0], atol=0.0001), 'no coincide con LogisticRegression(C=np.inf) de "
                                               "scikit-learn'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _w0, _w = newton_logistica(X_ent, y_ent, max_iter=0)\n'
                                               ">>> assert _w0 == 0 and np.allclose(_w, 0), 'con max_iter=0 debe devolver el punto inicial, w = 0'\n"
                                               '>>> _lam = 0.01\n'
                                               '>>> _sk = LogisticRegression(C=1 / (len(y_ent) * _lam), tol=1e-10, max_iter=1000).fit(X_ent, y_ent)\n'
                                               '>>> _w0, _w = newton_logistica(X_ent, y_ent, lam=_lam)\n'
                                               ">>> assert abs(_w0 - _sk.intercept_[0]) < 0.0001 and np.allclose(_w, _sk.coef_[0], atol=0.0001), 'con lam = 0.01 debe coincidir con "
                                               "LogisticRegression(C = 1/(N·lam))'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
