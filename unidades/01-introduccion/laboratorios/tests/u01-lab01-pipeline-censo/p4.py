OK_FORMAT = True

test = {   'name': 'p4',
    'points': 2,
    'suites': [   {   'cases': [   {'code': ">>> assert isinstance(preprocesamiento, ColumnTransformer), 'preprocesamiento debe ser un ColumnTransformer'\n", 'hidden': False, 'locked': False},
                                   {   'code': '>>> from sklearn.base import clone\n'
                                               '>>> _Z = clone(preprocesamiento).fit(X_tr).transform(X_te)\n'
                                               ">>> _Z = _Z.toarray() if hasattr(_Z, 'toarray') else np.asarray(_Z)\n"
                                               ">>> assert not np.isnan(_Z).any(), 'después de transformar no debe quedar ningún NaN'\n"
                                               ">>> assert 40 <= _Z.shape[1] <= 90, f'la matriz tiene {_Z.shape[1]} columnas; revisa el one-hot y min_frequency'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
