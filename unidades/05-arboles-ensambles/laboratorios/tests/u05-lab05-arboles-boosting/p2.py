OK_FORMAT = True

test = {   'name': 'p2',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': '>>> _u, _r = mejor_division(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 0.0, 10.0, 10.0]))\n'
                                               ">>> assert np.isclose(_u, 2.5) and np.isclose(_r, 100.0), 'con y = (0, 0, 10, 10) se corta en 2.5 y la suma de cuadrados baja de 100 a 0'\n"
                                               '>>> _u, _r = mejor_division(np.array([3.0, 1.0, 4.0, 2.0]), np.array([10.0, 0.0, 10.0, 0.0]))\n'
                                               ">>> assert np.isclose(_u, 2.5) and np.isclose(_r, 100.0), 'los datos no vienen ordenados: hay que ordenarlos por x'\n"
                                               '>>> _u, _r = mejor_division(np.array([5.0, 5.0, 5.0]), np.array([1.0, 2.0, 3.0]))\n'
                                               ">>> assert np.isnan(_u) and _r == 0.0, 'si todos los x son iguales no hay división: (nan, 0.0)'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _u, _r = mejor_division(np.array([1.0, 1.0, 2.0, 2.0, 3.0]), np.array([0.0, 4.0, 0.0, 4.0, 9.0]))\n'
                                               ">>> assert np.isclose(_u, 2.5), 'no se puede cortar entre dos x iguales: el único umbral bueno aquí es 2.5'\n"
                                               '>>> _u, _r = mejor_division(np.arange(6.0), np.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0]), min_hoja=2)\n'
                                               ">>> assert np.isclose(_u, 1.5), 'con min_hoja=2 cada lado debe tener al menos dos ejemplos'\n"
                                               '>>> _t = DecisionTreeRegressor(max_depth=1).fit(X_ent[:, [0]], y_ent)\n'
                                               '>>> _u, _r = mejor_division(X_ent[:, 0], y_ent)\n'
                                               ">>> assert np.isclose(_u, _t.tree_.threshold[0]), 'con la hora, el umbral debe coincidir con el de DecisionTreeRegressor(max_depth=1)'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
