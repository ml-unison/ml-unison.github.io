OK_FORMAT = True

test = {   'name': 'p1',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': '>>> _P = particiones_origen_movil(10, inicial=6, horizonte=2)\n'
                                               ">>> assert isinstance(_P, list) and len(_P) == 3, 'con n = 10, inicial = 6 y horizonte = 2 los orígenes son 6, 7 y 8'\n"
                                               ">>> assert np.array_equal(_P[0][0], np.arange(6)) and np.array_equal(_P[0][1], [6, 7]), 'el primer origen entrena con 0..5 y prueba en 6, 7'\n"
                                               ">>> assert np.array_equal(_P[-1][1], [8, 9]), 'la última prueba debe terminar en el último índice, 9'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _P = particiones_origen_movil(10, inicial=6, horizonte=2, ventana=3)\n'
                                               ">>> assert np.array_equal(_P[0][0], [3, 4, 5]) and np.array_equal(_P[2][0], [5, 6, 7]), 'con ventana = 3 se entrena solo con los 3 índices anteriores "
                                               "al origen'\n"
                                               ">>> assert particiones_origen_movil(10, inicial=9, horizonte=2) == [], 'si la prueba no cabe, no hay particiones'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
