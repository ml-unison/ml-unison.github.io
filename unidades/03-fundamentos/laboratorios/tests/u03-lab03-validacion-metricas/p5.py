OK_FORMAT = True

test = {   'name': 'p5',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _fpr, _tpr = curva_roc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))\n'
                                               ">>> assert np.allclose(_fpr, [0, 0, 0.5, 0.5, 1]) and np.allclose(_tpr, [0, 0.5, 0.5, 1, 1]), 'para este ejemplo, fpr = [0, 0, .5, .5, 1] y tpr = [0, "
                                               ".5, .5, 1, 1]'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _fpr, _tpr = curva_roc(y_pru, prob_logistica)\n'
                                               '>>> _f, _t, _ = roc_curve(y_pru, prob_logistica, drop_intermediate=False)\n'
                                               ">>> assert len(_fpr) == len(_f), 'debe haber un punto por cada valor distinto del puntaje, más el (0, 0)'\n"
                                               ">>> assert np.allclose(_fpr, _f) and np.allclose(_tpr, _t), 'no coincide con roc_curve de scikit-learn'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
