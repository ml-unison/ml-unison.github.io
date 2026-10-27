OK_FORMAT = True

test = {   'name': 'p5',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _X, _o = matriz_rezagos(np.arange(10.0), [1, 3])\n'
                                               ">>> assert np.shape(_X) == (7, 2) and np.shape(_o) == (7,), 'con T = 10 y rezagos (1, 3) hay 10 − 3 = 7 renglones'\n"
                                               ">>> assert np.array_equal(_o, np.arange(3.0, 10.0)), 'el objetivo empieza en y_3'\n"
                                               ">>> assert np.array_equal(_X[0], [2.0, 0.0]) and np.array_equal(_X[-1], [8.0, 6.0]), 'el renglón de y_3 tiene (y_2, y_0) y el de y_9 tiene (y_8, "
                                               "y_6)'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
