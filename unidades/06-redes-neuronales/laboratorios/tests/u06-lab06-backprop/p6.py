OK_FORMAT = True

test = {   'name': 'p6',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert len(historia) == 30, 'historia tiene un valor por época'\n"
                                               ">>> assert historia[-1] < 0.1 * historia[0] + 0.05, 'la pérdida de entrenamiento debe bajar mucho a lo largo de las épocas'\n"
                                               ">>> assert exactitud_np >= 0.95, f'la red debe acertar al menos 0.95 en prueba (obtuviste {exactitud_np:.3f})'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
