OK_FORMAT = True

test = {   'name': 'p5',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _V, _p, _Z = pca(X, 2)\n'
                                               ">>> assert _V.shape == (2, 2) and _p.shape == (2,) and (_Z.shape == (600, 2)), 'revisa las formas de lo que devuelve pca'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _rng = np.random.default_rng(0)\n'
                                               '>>> _A = _rng.normal(size=(200, 5)) @ _rng.normal(size=(5, 5))\n'
                                               '>>> _V, _p, _Z = pca(_A, 3)\n'
                                               '>>> _sk = PCA(n_components=3).fit(_A)\n'
                                               ">>> assert np.allclose(_p, _sk.explained_variance_ratio_), 'la proporción de varianza no coincide con la de scikit-learn'\n"
                                               ">>> assert np.allclose(np.abs(_V), np.abs(_sk.components_), atol=1e-06), 'las componentes no coinciden con las de scikit-learn (salvo signo)'\n"
                                               ">>> assert np.allclose(np.abs(_Z), np.abs(_sk.transform(_A)), atol=1e-06), 'la proyección no coincide con la de scikit-learn (salvo signo)'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
