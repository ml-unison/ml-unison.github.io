OK_FORMAT = True

test = {   'name': 'p7',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': '>>> _u, _c = umbral_costo(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), 1.0, 1.0)\n'
                                               ">>> assert abs(_u - 0.35) < 1e-12 and _c == 1.0, 'con costos iguales, en este ejemplo el mejor umbral es 0.35 con costo 1'\n"
                                               '>>> _u, _c = umbral_costo(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), 10.0, 1.0)\n'
                                               ">>> assert _u == 0.8 and _c == 1.0, 'si los falsos positivos son caros, conviene marcar solo al 0.8 (costo 1)'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {'code': ">>> assert 0.05 < u_opt < 0.5, 'con c_fn = 5 · c_fp, el umbral debería quedar bastante por debajo de 0.5'\n", 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
