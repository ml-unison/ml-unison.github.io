OK_FORMAT = True

test = {   'name': 'p2',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert abs(entropia_cruzada(w_cero, A_ent, y_ent) - np.log(2)) < 1e-12, 'con w = 0 todas las probabilidades son 1/2 y la entropía cruzada media es "
                                               "log 2'\n"
                                               ">>> assert np.shape(gradiente(w_cero, A_ent, y_ent)) == (6,), 'el gradiente tiene una entrada por columna de X'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _rng = np.random.default_rng(1)\n'
                                               '>>> _A = con_unos(_rng.normal(size=(40, 3)))\n'
                                               '>>> _y = (_rng.uniform(size=40) < 0.4).astype(float)\n'
                                               '>>> _w = _rng.normal(size=4)\n'
                                               '>>> _h = 1e-06\n'
                                               '>>> _dif = np.array([(entropia_cruzada(_w + _h * _e, _A, _y) - entropia_cruzada(_w - _h * _e, _A, _y)) / (2 * _h) for _e in np.eye(4)])\n'
                                               ">>> assert np.allclose(gradiente(_w, _A, _y), _dif, atol=1e-06), 'el gradiente no coincide con las diferencias finitas de tu entropía cruzada (¿el "
                                               "signo? ¿dividiste entre N en las dos?)'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
