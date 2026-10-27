OK_FORMAT = True

test = {   'name': 'p8',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': '>>> _m = RedConvolucional()\n'
                                               ">>> assert tuple(_m(torch.zeros(5, 1, 8, 8)).shape) == (5, 10), 'la salida debe ser de N × 10'\n"
                                               ">>> assert sum((p.numel() for p in _m.parameters())) == num_parametros, 'tu num_parametros no coincide con el conteo de PyTorch (o la arquitectura no "
                                               "es la de la tabla)'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
