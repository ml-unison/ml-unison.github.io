OK_FORMAT = True

test = {   'name': 'p4',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': ">>> assert C.shape == (5, 2) and etiquetas.shape == (600,), 'revisa las formas de C y de las etiquetas'\n"
                                               ">>> assert np.array_equal(etiquetas, asignar(X, C)), 'las etiquetas deben corresponder a los centroides finales'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> assert len(historial) >= 2, 'el historial debe tener una inercia por iteración'\n"
                                               ">>> assert all((b <= a + 1e-09 for a, b in zip(historial, historial[1:]))), 'la inercia nunca debe crecer de una iteración a la siguiente'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _mejor = min((kmedias(X, 5, semilla=s)[2][-1] for s in range(10)))\n'
                                               '>>> _sk = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X).inertia_\n'
                                               ">>> assert abs(_mejor - _sk) / _sk < 0.01, f'la mejor de 10 semillas ({_mejor:.1f}) debería igualar a scikit-learn ({_sk:.1f})'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
