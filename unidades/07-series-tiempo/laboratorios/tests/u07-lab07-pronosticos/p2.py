OK_FORMAT = True

test = {   'name': 'p2',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.array_equal(pronostico_ingenuo(np.array([3.0, 1.0, 4.0]), 3), [4.0, 4.0, 4.0]), 'el ingenuo repite el último valor'\n"
                                               ">>> assert np.array_equal(pronostico_estacional(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2, m=2), [4.0, 5.0]), 'con m = 2 el pronóstico a 1 y 2 pasos son "
                                               "los dos últimos valores'\n"
                                               ">>> assert np.array_equal(pronostico_estacional(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 5, m=2), [4.0, 5.0, 4.0, 5.0, 4.0]), 'con h > m se repite el "
                                               "último ciclo'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
