OK_FORMAT = True

test = {   'name': 'p3',
    'points': 4,
    'suites': [   {   'cases': [   {   'code': '>>> _a = ajustar_arbol(X_ent, y_ent, max_depth=0)\n'
                                               ">>> assert 'valor' in _a and np.isclose(_a['valor'], y_ent.mean()), 'con max_depth=0 el árbol es una hoja con la media'\n"
                                               '>>> _a = ajustar_arbol(X_ent, y_ent, max_depth=1)\n'
                                               ">>> assert _a.get('atributo') == 0 and np.isclose(_a['umbral'], 6.5), 'la primera pregunta es por la hora: ≤ 6.5'\n"
                                               ">>> assert np.shape(predecir_arbol(_a, X_pru)) == (len(X_pru),), 'una predicción por renglón'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> _a = ajustar_arbol(X_ent, y_ent, max_depth=4, min_samples_leaf=5)\n'
                                               '>>> _sk = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=0).fit(X_ent, y_ent)\n'
                                               ">>> assert np.allclose(predecir_arbol(_a, X_pru), _sk.predict(X_pru)), 'con max_depth=4 y min_samples_leaf=5 tus predicciones deben coincidir con las "
                                               "de DecisionTreeRegressor'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
