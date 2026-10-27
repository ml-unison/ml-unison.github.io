OK_FORMAT = True

test = {   'name': 'p5',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': '>>> _p = iniciar_parametros(64, 8, 10, semilla=1)\n'
                                               '>>> _E, _g = adelante_atras(_p, X_ent[:20], y_ent[:20])\n'
                                               ">>> assert isinstance(_E, float) and set(_g) == {'W1', 'b1', 'W2', 'b2'}, 'grads es un diccionario con llaves W1, b1, W2 y b2'\n"
                                               ">>> assert all((_g[k].shape == _p[k].shape for k in _p)), 'cada gradiente tiene la forma de su parámetro'\n"
                                               ">>> for _k in ['b2', 'W2', 'b1']:\n"
                                               '...     _num = gradiente_numerico(lambda v: adelante_atras({**_p, _k: v}, X_ent[:20], y_ent[:20])[0], _p[_k].copy())\n'
                                               "...     assert error_relativo(_g[_k], _num) < 1e-06, f'el gradiente de {_k} no coincide con las diferencias finitas'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
