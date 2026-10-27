OK_FORMAT = True

test = {   'name': 'p6',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.isclose(peso_hoja(-6.0, 2.0, lam=1.0), 2.0) and np.isclose(peso_hoja(4.0, 3.0, lam=0.0), -4 / 3), 'w* = −G/(H + λ)'\n"
                                               ">>> assert np.allclose(peso_hoja(np.array([1.0, -2.0]), np.array([1.0, 3.0]), lam=1.0), [-0.5, 0.5]), 'debe estar vectorizada'\n"
                                               ">>> assert np.isclose(ganancia_xgb(-4.0, 2.0, 4.0, 2.0, lam=0.0, gamma=0.0), 8.0), '½[16/2 + 16/2 − 0/4] = 8'\n"
                                               ">>> assert np.isclose(ganancia_xgb(-4.0, 2.0, 4.0, 2.0, lam=2.0, gamma=1.0), 3.0), '½[16/4 + 16/4 − 0/6] − 1 = 3'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _u, _g = mejor_division_xgb(X_ent[:, 0], g_bicis, h_bicis, lam=0.0)\n'
                                               '>>> _ud, _rd = mejor_division(X_ent[:, 0], y_ent)\n'
                                               ">>> assert np.isclose(_u, _ud) and np.isclose(_g, _rd / 2), 'con pérdida cuadrática, h = 1 y λ = γ = 0, la ganancia es la mitad de la reducción de la "
                                               "suma de cuadrados'\n"
                                               '>>> _u, _g = mejor_division_xgb(np.array([2.0, 2.0]), np.array([1.0, -1.0]), np.ones(2))\n'
                                               ">>> assert np.isnan(_u) and _g == 0.0, 'sin valores distintos de x no hay división'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
