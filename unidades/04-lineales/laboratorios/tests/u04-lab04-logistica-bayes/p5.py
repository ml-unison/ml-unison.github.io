OK_FORMAT = True

test = {   'name': 'p5',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': '>>> _lp, _lt = ajustar_nb(C_ent, t_ent, alpha=1.0)\n'
                                               ">>> assert np.shape(_lp) == (2,) and np.shape(_lt) == (2, C_ent.shape[1]), 'log_previas es de longitud 2 y log_theta de 2 × V'\n"
                                               ">>> assert abs(np.exp(_lp).sum() - 1) < 1e-12 and abs(np.exp(_lp[1]) - t_ent.mean()) < 1e-12, 'las previas (en logaritmo) son las proporciones de cada "
                                               "clase'\n"
                                               ">>> assert np.allclose(ajustar_nb(sparse.csr_matrix(np.eye(3)), np.array([0, 1, 1]))[0], np.log([1 / 3, 2 / 3])), 'con etiquetas (0, 1, 1) las previas "
                                               "son log(1/3) y log(2/3)'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _lp, _lt = ajustar_nb(C_ent, t_ent, alpha=1.0)\n'
                                               ">>> assert np.allclose(np.exp(_lt).sum(axis=1), 1.0), 'cada renglón de θ debe sumar 1 (¿falta α·V en el denominador?)'\n"
                                               '>>> _X = sparse.csr_matrix(np.array([[2, 0, 1], [0, 1, 0], [1, 1, 0]]))\n'
                                               '>>> _lp, _lt = ajustar_nb(_X, np.array([0, 1, 1]), alpha=1.0)\n'
                                               ">>> assert np.allclose(np.exp(_lt), [[3 / 6, 1 / 6, 2 / 6], [2 / 6, 3 / 6, 1 / 6]]), 'en este ejemplo θ₀ = (3, 1, 2)/6 y θ₁ = (2, 3, 1)/6'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
