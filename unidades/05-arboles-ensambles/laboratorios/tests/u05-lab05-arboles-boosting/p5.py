OK_FORMAT = True

test = {   'name': 'p5',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _f0, _arb = ajustar_gb(X_ent, y_ent, n_arboles=1, tasa=0.5, max_depth=2)\n'
                                               ">>> assert isinstance(_f0, float) and np.isclose(_f0, y_ent.mean()) and (len(_arb) == 1), 'f0 es la media y hay un árbol por iteración'\n"
                                               '>>> _h = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_ent, y_ent - y_ent.mean())\n'
                                               ">>> assert np.allclose(_arb[0].predict(X_pru), _h.predict(X_pru)), 'el primer árbol se ajusta a los residuos y − ȳ'\n"
                                               '>>> assert np.allclose(predecir_gb(_f0, _arb, X_pru, tasa=0.5), y_ent.mean() + 0.5 * _h.predict(X_pru))\n',
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _rng = np.random.default_rng(8)\n'
                                               '>>> _X = _rng.uniform(-2, 2, size=(400, 3))\n'
                                               '>>> _y = np.sin(2 * _X[:, 0]) * _X[:, 1] + _X[:, 2] + 0.2 * _rng.normal(size=400)\n'
                                               '>>> _sk = GradientBoostingRegressor(n_estimators=40, learning_rate=0.2, max_depth=3, random_state=0).fit(_X, _y)\n'
                                               '>>> _f0, _arb = ajustar_gb(_X, _y, n_arboles=40, tasa=0.2, max_depth=3)\n'
                                               ">>> assert np.allclose(predecir_gb(_f0, _arb, _X, tasa=0.2), _sk.predict(_X), atol=1e-06), 'tu gradient boosting debe coincidir con "
                                               "GradientBoostingRegressor(n_estimators=40, learning_rate=0.2, max_depth=3)'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
