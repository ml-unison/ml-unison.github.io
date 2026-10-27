OK_FORMAT = True

test = {   'name': 'p6',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _J = log_conjunta(C_pru, log_previas, log_theta)\n'
                                               ">>> assert np.shape(_J) == (C_pru.shape[0], 2), 'J es de N × 2'\n"
                                               '>>> _x = C_pru[0]\n'
                                               ">>> assert np.isclose(_J[0, 1], log_previas[1] + sum((c * log_theta[1, j] for j, c in zip(_x.indices, _x.data)))), 'J[n, c] = log π_c + Σ_j x_nj · log "
                                               "θ_cj'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _P = proba_nb(C_pru, log_previas, log_theta)\n'
                                               '>>> _J = log_conjunta(C_pru, log_previas, log_theta)\n'
                                               ">>> assert np.shape(_P) == (C_pru.shape[0], 2) and np.allclose(_P.sum(axis=1), 1.0), 'cada renglón de probabilidades debe sumar 1'\n"
                                               ">>> assert np.all(np.argmax(_P, axis=1) == np.argmax(_J, axis=1)), 'la clase más probable es la de mayor J'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
