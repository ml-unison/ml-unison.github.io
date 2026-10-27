OK_FORMAT = True

test = {   'name': 'p6',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert abs(auc_pares(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) - 0.75) < 1e-12, 'en este ejemplo 3 de los 4 pares (positivo, "
                                               "negativo) están bien ordenados'\n"
                                               ">>> assert abs(auc_pares(np.array([0, 1]), np.array([0.5, 0.5])) - 0.5) < 1e-12, 'un empate vale 1/2'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> assert abs(auc_logistica - roc_auc_score(y_pru, prob_logistica)) < 1e-09, 'auc_logistica no coincide con roc_auc_score'\n"
                                               ">>> assert abs(auc_ridge - roc_auc_score(y_pru, puntaje_ridge)) < 1e-09, 'auc_ridge no coincide con roc_auc_score'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
