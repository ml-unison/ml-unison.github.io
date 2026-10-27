OK_FORMAT = True

test = {   'name': 'p4',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': '>>> _arb, _mue = ajustar_bagging(X_ent[:500], y_ent[:500], B=7, semilla=3, max_depth=2)\n'
                                               ">>> assert len(_arb) == 7 and np.shape(_mue) == (7, 500), 'B árboles y una remuestra de N índices por árbol'\n"
                                               ">>> assert np.array_equal(_mue, np.random.default_rng(3).integers(0, 500, size=(7, 500))), 'las remuestras deben generarse con rng.integers(0, N, "
                                               "size=(B, N)) y rng = np.random.default_rng(semilla)'\n"
                                               '>>> assert np.allclose(predecir_bagging(_arb, X_pru[:20]), np.mean([a.predict(X_pru[:20]) for a in _arb], axis=0))\n',
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _X = np.arange(6.0)[:, None]\n'
                                               '>>> _mue = np.array([[0, 0, 1, 1, 2, 5], [3, 3, 4, 4, 5, 5], [0, 1, 2, 3, 4, 5]])\n'
                                               '>>> _arb = [DecisionTreeRegressor().fit(_X[m], 10.0 * (_k + 1) * np.ones(6)) for _k, m in enumerate(_mue)]\n'
                                               '>>> _o = prediccion_oob(_arb, _mue, _X)\n'
                                               ">>> assert np.allclose(_o[:5], [20, 20, 20, 10, 10]) and np.isnan(_o[5]), 'OOB del ejemplo i: promedio de los árboles cuya remuestra no contiene i "
                                               "(nan si todas lo contienen)'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
