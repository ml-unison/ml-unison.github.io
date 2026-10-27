OK_FORMAT = True

test = {   'name': 'p7',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.allclose(pronosticar_ar(np.array([0.0, 2.0]), 1.0, np.array([0.5]), 3), [2.0, 2.0, 2.0]), 'AR(1) con c = 1, φ = 0.5 y y_T = 2 (su "
                                               "media): el pronóstico se queda en 2'\n"
                                               ">>> assert np.allclose(pronosticar_ar(np.array([1.0, 3.0]), 0.0, np.array([0.5, 0.25]), 2), [1.75, 1.625]), 'con c = 0, φ = (0.5, 0.25): ŷ₁ = 0.5·3 + "
                                               "0.25·1 = 1.75 y ŷ₂ = 0.5·1.75 + 0.25·3 = 1.625'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
