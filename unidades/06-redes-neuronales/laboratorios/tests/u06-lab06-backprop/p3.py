OK_FORMAT = True

test = {   'name': 'p3',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _E, _dZ = softmax_entropia(np.zeros((2, 4)), np.array([0, 3]))\n'
                                               ">>> assert abs(_E - np.log(4)) < 1e-12, 'con puntajes iguales cada clase tiene probabilidad 1/4 y la pérdida es log 4'\n"
                                               ">>> assert np.allclose(_dZ, [[-0.375, 0.125, 0.125, 0.125], [0.125, 0.125, 0.125, -0.375]]), 'dZ = (P − Y)/N: aquí (1/4 − 1)/2 = −0.375 en la clase "
                                               "correcta y (1/4)/2 = 0.125 en las otras'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _rng = np.random.default_rng(3)\n'
                                               '>>> _Z = _rng.normal(size=(5, 3))\n'
                                               '>>> _y = np.array([0, 2, 1, 1, 0])\n'
                                               '>>> _h = 1e-06\n'
                                               '>>> _dif = np.zeros_like(_Z)\n'
                                               '>>> for _i in range(5):\n'
                                               '...     for _j in range(3):\n'
                                               '...         _Zm, _Zp = (_Z.copy(), _Z.copy())\n'
                                               '...         _Zm[_i, _j] -= _h\n'
                                               '...         _Zp[_i, _j] += _h\n'
                                               '...         _dif[_i, _j] = (softmax_entropia(_Zp, _y)[0] - softmax_entropia(_Zm, _y)[0]) / (2 * _h)\n'
                                               ">>> assert np.allclose(softmax_entropia(_Z, _y)[1], _dif, atol=1e-07), 'dZ no coincide con las diferencias finitas de tu pérdida (¿dividiste entre "
                                               "N?)'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
