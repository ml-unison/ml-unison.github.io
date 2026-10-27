OK_FORMAT = True

test = {   'name': 'p4',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.allclose(acf_muestral(np.array([1.0, 2.0, 3.0, 4.0]), 3), [1.0, 0.25, -0.3, -0.45]), 'para (1, 2, 3, 4): la serie centrada es (−1.5, "
                                               "−0.5, 0.5, 1.5), Σz² = 5 y r = (1, 0.25, −0.3, −0.45)'\n"
                                               '>>> _r = acf_muestral(anomalia[:corte], 10)\n'
                                               ">>> assert np.shape(_r) == (11,) and abs(_r[0] - 1) < 1e-12, 'devuelve k_max + 1 valores y r_0 = 1'\n"
                                               ">>> assert np.allclose(_r, acf(anomalia[:corte], nlags=10, fft=False)), 'no coincide con statsmodels.tsa.stattools.acf'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
