OK_FORMAT = True

test = {   'name': 'p6',
    'points': 1,
    'suites': [   {   'cases': [   {'code': ">>> assert isinstance(n_90, (int, np.integer)) and 1 <= n_90 <= 64, 'n_90 debe ser un entero entre 1 y 64'\n", 'hidden': False, 'locked': False},
                                   {   'code': '>>> _acum = np.cumsum(PCA().fit(digitos).explained_variance_ratio_)\n'
                                               ">>> assert _acum[n_90 - 1] >= 0.9, 'con n_90 componentes todavía no llegas al 90 %'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
