OK_FORMAT = True

test = {   'name': 'p3',
    'points': 1,
    'suites': [   {   'cases': [   {   'code': ">>> assert set(numericas) == {'age', 'education-num', 'hours-per-week'}, 'revisa la lista numericas'\n"
                                               ">>> assert set(sesgadas) == {'capital-gain', 'capital-loss'}, 'revisa la lista sesgadas'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> assert 'education' not in nominales and 'fnlwgt' not in nominales, 'education y fnlwgt no van en nominales'\n"
                                               ">>> assert len(nominales) == 7, 'nominales debe tener siete columnas'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
