OK_FORMAT = True

test = {   'name': 'p4',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.shape(errores_cv) == (29,) and np.shape(desv_cv) == (29,), 'debe haber un error y una desviación por cada lambda de la rejilla'\n"
                                               ">>> assert lam_min in rejilla and lam_1de in rejilla, 'los lambdas elegidos deben ser valores de la rejilla'\n"
                                               ">>> assert lam_1de >= lam_min, 'la regla de una desviación estándar nunca elige un modelo menos regularizado que el mínimo'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> assert abs(errores_cv[0] - validacion_cruzada(ajustar_ridge(rejilla[0]), X_ent, y_ent, 10, 0, ecm)[0]) < 1e-12, 'errores_cv[0] debe ser la "
                                               "validación cruzada con el primer lambda, k=10 y semilla=0'\n"
                                               ">>> assert errores_cv[np.searchsorted(rejilla, lam_min)] == errores_cv.min(), 'lam_min debe tener el menor error medio'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
