OK_FORMAT = True

test = {   'name': 'p2',
    'points': 1,
    'suites': [   {   'cases': [   {'code': ">>> assert len(X_tr) == 9600 and len(X_te) == 2400, 'la prueba debe ser el 20 % de los datos'\n", 'hidden': False, 'locked': False},
                                   {   'code': ">>> assert 'ingreso' not in X_tr.columns, 'X no debe incluir la variable objetivo'\n"
                                               ">>> assert set(np.unique(y_tr)) == {0, 1}, 'y debe tener valores 0 y 1'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
