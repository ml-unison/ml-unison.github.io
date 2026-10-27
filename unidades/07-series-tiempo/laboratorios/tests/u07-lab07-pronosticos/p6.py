OK_FORMAT = True

test = {   'name': 'p6',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _c, _phi = ajustar_ar(anomalia[:corte], 2)\n'
                                               ">>> assert np.shape(_phi) == (2,), 'phi tiene p coeficientes'\n"
                                               ">>> _sm = AutoReg(anomalia[:corte], lags=2, trend='c').fit().params\n"
                                               '>>> assert abs(_c - _sm[0]) < 1e-08 and np.allclose(_phi, _sm[1:]), "no coincide con AutoReg(lags=2, trend=\'c\') de statsmodels"\n',
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
