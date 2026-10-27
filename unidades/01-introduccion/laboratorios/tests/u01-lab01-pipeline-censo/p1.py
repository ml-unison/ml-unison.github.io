OK_FORMAT = True

test = {   'name': 'p1',
    'points': 1,
    'suites': [   {   'cases': [   {'code': ">>> assert censo.shape == (12000, 15), 'censo debe tener 12,000 filas y 15 columnas'\n", 'hidden': False, 'locked': False},
                                   {'code': '>>> assert censo[\'occupation\'].isna().sum() > 0, "los \'?\' de occupation deben quedar como faltantes (NaN)"\n', 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
