OK_FORMAT = True

test = {   'name': 'p7',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert muestra.shape == (5000, 3), 'la muestra debe tener 5,000 píxeles'\n"
                                               ">>> assert colores.shape == (16, 3), 'colores debe tener 16 filas (una por color) y 3 columnas'\n"
                                               ">>> assert cuantizada.shape == imagen.shape, 'cuantizada debe tener la misma forma que imagen'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> assert len(np.unique(cuantizada.reshape(-1, 3), axis=0)) <= 16, 'la imagen cuantizada debe usar a lo más 16 colores'\n"
                                               ">>> assert 0 < ecm < 0.003, f'el error cuadrático medio ({ecm:.5f}) es demasiado grande'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
