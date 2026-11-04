OK_FORMAT = True

test = {   'name': 'p6',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert len(historia) == 30 and isinstance(historia[0], float), 'historia tiene un float por época'\n"
                                               ">>> assert historia[-1] < 0.1 * historia[0] + 0.05, 'la pérdida de entrenamiento debe bajar mucho a lo largo de las épocas'\n"
                                               ">>> assert exactitud_a >= 0.95, f'la red debe acertar al menos 0.95 en prueba (obtuviste {exactitud_a:.3f})'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
