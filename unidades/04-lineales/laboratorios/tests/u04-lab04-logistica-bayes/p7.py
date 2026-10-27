OK_FORMAT = True

test = {   'name': 'p7',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.shape(prob_spam) == (len(t_pru),) and np.shape(pred_nb) == (len(t_pru),), 'debe haber un valor por mensaje de prueba'\n"
                                               ">>> assert abs(exactitud_nb - 0.9842) < 5e-05, 'con α = 1 la exactitud en prueba debe ser 0.9842'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
