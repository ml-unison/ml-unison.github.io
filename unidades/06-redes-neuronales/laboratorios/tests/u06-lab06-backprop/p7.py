OK_FORMAT = True

test = {   'name': 'p7',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert isinstance(perdida_mlp, float) and perdida_mlp < 0.2, 'la pérdida de la última época debe ser un float pequeño'\n"
                                               ">>> assert exactitud_mlp >= 0.95, 'el MLP de PyTorch debe acertar al menos 0.95 en prueba'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
