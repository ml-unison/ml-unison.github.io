OK_FORMAT = True

test = {   'name': 'p1',
    'points': 1,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.allclose(sigmoide(np.array([-2.0, 0.0, 2.0])), [0.11920292, 0.5, 0.88079708]), 'σ(0) = 0.5 y σ(±2) ≈ 0.1192, 0.8808'\n"
                                               ">>> assert sigmoide(np.zeros((3, 4))).shape == (3, 4), 'la salida debe tener la misma forma que la entrada'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> with np.errstate(over='raise', invalid='raise'):\n"
                                               '...     _s = sigmoide(np.array([-1000.0, 1000.0]))\n'
                                               ">>> assert np.allclose(_s, [0.0, 1.0]), 'σ(−1000) = 0 y σ(1000) = 1, sin avisos de desbordamiento'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
