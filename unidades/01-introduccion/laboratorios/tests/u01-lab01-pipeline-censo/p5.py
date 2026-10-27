OK_FORMAT = True

test = {   'name': 'p5',
    'points': 2,
    'suites': [   {   'cases': [   {'code': ">>> assert abs(exactitud_base - 0.761) < 0.005, 'la línea base debe predecir siempre la clase más frecuente'\n", 'hidden': False, 'locked': False},
                                   {'code': ">>> assert 0.82 < exactitud_modelo < 0.9, f'exactitud del modelo fuera de lo esperado: {exactitud_modelo:.3f}'\n", 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
