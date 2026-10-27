OK_FORMAT = True

test = {   'name': 'p3',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _m, _s = validacion_cruzada(ajustar_media, X_ent, y_ent, k=10, semilla=0)\n'
                                               ">>> assert abs(_m - y_ent.var()) < 0.002, 'para el modelo constante, el ECM de validación debe parecerse a la varianza de y'\n"
                                               ">>> assert 0 < _s < 0.05, 'la desviación estándar de los errores debe ser positiva y pequeña'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _X = np.arange(6.0).reshape(-1, 1)\n'
                                               '>>> _y = np.array([0.0, 0.0, 0.0, 6.0, 6.0, 6.0])\n'
                                               '>>> _m, _s = validacion_cruzada(ajustar_media, _X, _y, k=6, semilla=0)\n'
                                               ">>> assert abs(_m - 12.96) < 1e-09 and abs(_s) < 1e-09, 'dejando uno fuera en este ejemplo, cada error es 3.6² = 12.96: la media debe ser 12.96 y la "
                                               "desviación 0'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
