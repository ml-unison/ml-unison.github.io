OK_FORMAT = True

test = {   'name': 'p1',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.allclose(gini(ejemplo), [0.5, 0.18, 0.0, 0.0]), 'Gini de (50, 50), (90, 10), (100, 0) y (0, 0): 0.5, 0.18, 0 y 0'\n"
                                               ">>> assert np.allclose(entropia(ejemplo), [1.0, 0.4689956, 0.0, 0.0]), 'entropía en bits: 1, 0.469, 0 y 0'\n"
                                               ">>> assert np.shape(gini(ejemplo)) == (4,) and np.shape(entropia(ejemplo)) == (4,), 'un valor por renglón'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> with np.errstate(all='raise'):\n"
                                               '...     _g, _h = (gini(np.array([[0, 0, 0], [3, 0, 0]])), entropia(np.array([[0, 0, 0], [3, 0, 0]])))\n'
                                               ">>> assert np.allclose(_g, 0) and np.allclose(_h, 0), 'nodos vacíos o puros: impureza 0, sin avisos de numpy'\n"
                                               ">>> assert np.isclose(entropia(np.array([1, 1, 1, 1])), 2.0), 'cuatro clases parejas: 2 bits (también con un solo nodo, de forma (K,))'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
