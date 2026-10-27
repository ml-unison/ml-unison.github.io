OK_FORMAT = True

test = {   'name': 'p3',
    'points': 1,
    'suites': [   {   'cases': [   {   'code': '>>> _H = hessiana(w_cero, A_ent)\n'
                                               ">>> assert _H.shape == (6, 6), 'la hessiana es de (d+1) × (d+1)'\n"
                                               ">>> assert np.allclose(_H, A_ent.T @ A_ent / (4 * len(A_ent))), 'con w = 0, p(1 − p) = 1/4 para todos y la hessiana es XᵀX / (4N)'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _rng = np.random.default_rng(2)\n'
                                               '>>> _A = con_unos(_rng.normal(size=(50, 2)))\n'
                                               '>>> _y = (_rng.uniform(size=50) < 0.5).astype(float)\n'
                                               '>>> _w = np.array([0.2, -0.7, 1.1])\n'
                                               '>>> _h = 1e-06\n'
                                               '>>> _dif = np.column_stack([(gradiente(_w + _h * _e, _A, _y) - gradiente(_w - _h * _e, _A, _y)) / (2 * _h) for _e in np.eye(3)])\n'
                                               ">>> assert np.allclose(hessiana(_w, _A), _dif, atol=1e-06), 'la hessiana no coincide con las diferencias finitas de tu gradiente'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
